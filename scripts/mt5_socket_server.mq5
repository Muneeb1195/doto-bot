//+------------------------------------------------------------------+
//| MT5 Socket Server — exposes MT5 API over TCP to native Linux bot  |
//| Runs as Expert Advisor inside MT5 terminal under Wine             |
//+------------------------------------------------------------------+
#property copyright "Doto Bot"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>
#include <Trade\DealInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>

// --- Winsock structures (must precede #import) ---
struct sockaddr_in
{
   short  sin_family;
   ushort sin_port;
   uint   sin_addr;
   char   sin_zero[8];
};

// --- Winsock API via DLL import ---
// NOTE: MQL5's built-in SocketCreate() produces a terminal-internal handle that
// is NOT a Winsock SOCKET, and MQL5 has no server-socket API at all. So the
// entire socket layer here is raw Winsock. SOCKET is 64-bit on Win64 -> long.
// Requires "Allow DLL imports" enabled in the EA properties / terminal options.
#import "ws2_32.dll"
   long WSASocketW(int af, int type, int protocol, long lpProtocolInfo, int g, uint dwFlags);
   int  bind(long s, sockaddr_in &addr, int namelen);
   int  listen(long s, int backlog);
   long accept(long s, long addr, long addrlen);
   int  recv(long s, uchar &buf[], int len, int flags);
   int  send(long s, uchar &buf[], int len, int flags);
   int  closesocket(long s);
   int  WSAStartup(ushort wVersionRequested, uchar &lpWSAData[]);
   int  WSACleanup();
   int  WSAGetLastError();
   ushort htons(ushort hostshort);
   uint inet_addr(string cp);
   int  ioctlsocket(long s, int cmd, uint &argp);
#import

#define WS_INVALID_SOCKET  (-1)
#define WS_AF_INET         2
#define WS_SOCK_STREAM     1
#define WS_IPPROTO_TCP     6
#define WS_FIONBIO         (int)0x8004667E
#define WS_EWOULDBLOCK     10035

bool _wsaLoaded = false;

bool WsStartup()
{
   if(_wsaLoaded) return true;
   uchar wsadata[];
   ArrayResize(wsadata, 512);
   ArrayInitialize(wsadata, 0);
   int rc = WSAStartup((ushort)0x0202, wsadata); // MAKEWORD(2,2)
   if(rc != 0)
   {
      Print("[MT5Socket] WSAStartup failed: ", rc);
      return false;
   }
   _wsaLoaded = true;
   return true;
}

long WsCreate()
{
   return WSASocketW(WS_AF_INET, WS_SOCK_STREAM, WS_IPPROTO_TCP, 0, 0, 0);
}

void WsClose(long sock)
{
   if(sock != WS_INVALID_SOCKET) closesocket(sock);
}

bool WsSetNonBlocking(long sock)
{
   uint nb = 1;
   return (ioctlsocket(sock, WS_FIONBIO, nb) == 0);
}

// Parse a dotted-quad IPv4 string into network-byte-order uint.
// Do NOT use ws2_32!inet_addr: MQL5 passes strings as UTF-16, so the ANSI
// inet_addr sees "1\0" and returns INADDR_NONE -> bind() fails with 10049.
uint WsInetAddr(string address)
{
   string parts[];
   if(StringSplit(address, '.', parts) != 4)
      return 0xFFFFFFFF;
   uint result = 0;
   for(int i = 0; i < 4; i++)
   {
      int octet = (int)StringToInteger(parts[i]);
      if(octet < 0 || octet > 255)
         return 0xFFFFFFFF;
      result |= ((uint)octet) << (8 * i);   // network byte order on little-endian
   }
   return result;
}

bool WsBind(long sock, int port, string address)
{
   sockaddr_in addr;
   addr.sin_family = WS_AF_INET;
   addr.sin_port   = htons((ushort)port);
   addr.sin_addr   = WsInetAddr(address);
   for(int i = 0; i < 8; i++) addr.sin_zero[i] = 0;
   return (bind(sock, addr, sizeof(sockaddr_in)) == 0);
}

bool WsListen(long sock, int backlog)
{
   return (listen(sock, backlog) == 0);
}

long WsAccept(long sock)
{
   return accept(sock, 0, 0);
}

int WsRecv(long sock, uchar &array[], int count)
{
   if(ArraySize(array) < count) ArrayResize(array, count);
   return recv(sock, array, count, 0);
}

int WsSend(long sock, uchar &array[], int count)
{
   return send(sock, array, count, 0);
}

input int    SocketPort = 9000;
input int    PollIntervalMs = 50;
input int    MaxBufferSize = 65536;
input bool   UseTimer = true;
input int    StaleTimeoutSec = 120;

//--- globals
long   g_serverSocket = WS_INVALID_SOCKET;
long   g_clientSocket = WS_INVALID_SOCKET;
bool   g_initialized  = false;
bool   g_clientConnected = false;
string g_readBuffer   = "";
datetime g_lastDataTime = 0;
CTrade g_trade;
CPositionInfo g_posInfo;
COrderInfo g_orderInfo;
CDealInfo g_dealInfo;
CSymbolInfo g_symInfo;
CAccountInfo g_accInfo;

//--- constants mapping
#define CMD_INIT       "INIT"
#define CMD_SHUTDOWN   "SHUTDOWN"
#define CMD_ACCOUNT    "ACCOUNT"
#define CMD_TERMINAL   "TERMINAL"
#define CMD_VERSION    "VERSION"
#define CMD_LASTERR    "LASTERR"
#define CMD_SYMBOLS    "SYMBOLS"
#define CMD_SYMBOLS_TOTAL "SYMBOLS_TOTAL"
#define CMD_SYMBOL     "SYMBOL"
#define CMD_TICK       "TICK"
#define CMD_SELECT     "SELECT"
#define CMD_RATES_POS  "RATES_POS"
#define CMD_RATES_RANGE "RATES_RANGE"
#define CMD_ORDERS     "ORDERS"
#define CMD_POSITIONS  "POSITIONS"
#define CMD_ORDER_SEND "ORDER_SEND"
#define CMD_ORDER_CHECK "ORDER_CHECK"
#define CMD_HIST_ORDERS "HIST_ORDERS"
#define CMD_HIST_DEALS  "HIST_DEALS"
#define CMD_PING        "PING"

//+------------------------------------------------------------------+
int OnInit()
{
   Print("[MT5Socket] OnInit begin");

   if(!MQLInfoInteger(MQL_DLLS_ALLOWED))
   {
      Print("[MT5Socket] DLL imports are disabled. Enable 'Allow DLL imports' in the EA settings.");
      return INIT_FAILED;
   }

   Print("[MT5Socket] DLLs allowed, calling WSAStartup");
   if(!WsStartup()) return INIT_FAILED;

   Print("[MT5Socket] WSAStartup ok, creating socket");
   g_serverSocket = WsCreate();
   if(g_serverSocket == WS_INVALID_SOCKET)
   {
      Print("[MT5Socket] socket() failed: ", WSAGetLastError());
      return INIT_FAILED;
   }

   if(!WsBind(g_serverSocket, SocketPort, "127.0.0.1"))
   {
      Print("[MT5Socket] bind() failed: ", WSAGetLastError());
      WsClose(g_serverSocket);
      g_serverSocket = WS_INVALID_SOCKET;
      return INIT_FAILED;
   }

   if(!WsListen(g_serverSocket, 1))
   {
      Print("[MT5Socket] listen() failed: ", WSAGetLastError());
      WsClose(g_serverSocket);
      g_serverSocket = WS_INVALID_SOCKET;
      return INIT_FAILED;
   }

   if(!WsSetNonBlocking(g_serverSocket))
   {
      Print("[MT5Socket] ioctlsocket(FIONBIO) failed: ", WSAGetLastError());
      WsClose(g_serverSocket);
      g_serverSocket = WS_INVALID_SOCKET;
      return INIT_FAILED;
   }

    if(UseTimer) EventSetMillisecondTimer(PollIntervalMs);
    Print("[MT5Socket] Listening on 127.0.0.1:", SocketPort, "  (OnTick + UseTimer=", UseTimer, ")");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   WsClose(g_clientSocket);
   WsClose(g_serverSocket);
   g_clientSocket = WS_INVALID_SOCKET;
   g_serverSocket = WS_INVALID_SOCKET;
   if(_wsaLoaded) { WSACleanup(); _wsaLoaded = false; }
   Print("[MT5Socket] Stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
void PollSocket()
{
    //--- accept new connection if none
    if(!g_clientConnected)
    {
       long accepted = WsAccept(g_serverSocket);
       if(accepted != WS_INVALID_SOCKET)
       {
          WsSetNonBlocking(accepted);
          g_clientSocket = accepted;
          g_clientConnected = true;
          g_readBuffer = "";
          g_lastDataTime = TimeCurrent();
          Print("[MT5Socket] Client connected");
       }
       return;
    }

    //--- stale-client watchdog: drop silent connections so a stuck slot
    //    never blocks the single-client server indefinitely.
    if(TimeCurrent() - g_lastDataTime > StaleTimeoutSec)
    {
       Print("[MT5Socket] Client stale, closing (timeout)");
       WsClose(g_clientSocket);
       g_clientSocket = WS_INVALID_SOCKET;
       g_clientConnected = false;
       return;
    }

    //--- read available data
    uchar buf[];
    ArrayResize(buf, MaxBufferSize);
    int bytes = WsRecv(g_clientSocket, buf, MaxBufferSize);
    if(bytes < 0)
    {
       int err = WSAGetLastError();
       if(err != WS_EWOULDBLOCK)
       {
          Print("[MT5Socket] Client error, closing: ", err);
          WsClose(g_clientSocket);
          g_clientSocket = WS_INVALID_SOCKET;
          g_clientConnected = false;
       }
       return;
    }
    if(bytes == 0)
    {
       // graceful peer shutdown
       Print("[MT5Socket] Client disconnected");
       WsClose(g_clientSocket);
       g_clientSocket = WS_INVALID_SOCKET;
       g_clientConnected = false;
       return;
    }

    g_lastDataTime = TimeCurrent();

     //--- convert to string and append to buffer
     string chunk = CharArrayToString(buf, 0, bytes, CP_UTF8);
     g_readBuffer += chunk;

     //--- process complete lines
     string line;
     int pos;
     while((pos = StringFind(g_readBuffer, "\n")) >= 0)
     {
        line = StringSubstr(g_readBuffer, 0, pos);
        StringReplace(line, "\r", "");
        g_readBuffer = StringSubstr(g_readBuffer, pos + 1);
        if(StringLen(line) > 0) ProcessCommand(line);
     }
}

//+------------------------------------------------------------------+
//| OnTick — primary poll driver. Fires on every market tick from    |
//| MT5's internal engine (NOT the Wine message pump), so it works   |
//| reliably under headless Wine where OnTimer/WM_TIMER never fires. |
//| Throttled to PollIntervalMs to avoid spinning.                   |
//+------------------------------------------------------------------+
void OnTick()
{
    static datetime lastPoll = 0;
    datetime now = TimeCurrent();
    if(now - lastPoll < PollIntervalMs / 1000.0) return;
    lastPoll = now;
    PollSocket();
}

//+------------------------------------------------------------------+
//| OnTimer — kept as a harmless fallback. Disabled by default via   |
//| UseTimer=false because WM_TIMER does not dispatch under Wine.    |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(UseTimer) PollSocket();
}

//+------------------------------------------------------------------+
void SendResponse(string msg)
{
   if(!g_clientConnected || g_clientSocket == WS_INVALID_SOCKET) return;
   msg += "\n";
   uchar buf[];
   int len = StringToCharArray(msg, buf, 0, WHOLE_ARRAY, CP_UTF8) - 1; // drop NUL
   if(len <= 0) return;
   int sent = 0;
   while(sent < len)
   {
      uchar chunk[];
      int remaining = len - sent;
      ArrayResize(chunk, remaining);
      ArrayCopy(chunk, buf, 0, sent, remaining);
      int n = WsSend(g_clientSocket, chunk, remaining);
      if(n <= 0)
      {
         int err = WSAGetLastError();
         if(err == WS_EWOULDBLOCK) { Sleep(1); continue; }
         Print("[MT5Socket] send() failed: ", err);
         WsClose(g_clientSocket);
         g_clientSocket = WS_INVALID_SOCKET;
         g_clientConnected = false;
         return;
      }
      sent += n;
   }
}

//+------------------------------------------------------------------+
void SendOK(string data) { SendResponse("OK " + data); }
void SendERR(string msg) { SendResponse("ERR " + msg); }

//+------------------------------------------------------------------+
void ProcessCommand(string line)
{
   if(StringLen(line) == 0) return;

   string cmd;
   string args;
   int spacePos = StringFind(line, " ");
   if(spacePos > 0)
   {
      cmd  = StringSubstr(line, 0, spacePos);
      args = StringSubstr(line, spacePos + 1);
   }
   else
   {
      cmd = line;
      args = "";
   }

   StringToUpper(cmd);

   if(cmd == CMD_PING)              { SendOK("pong"); return; }
   if(cmd == CMD_INIT)              { CmdInit(args); return; }
   if(cmd == CMD_SHUTDOWN)          { CmdShutdown(); return; }
   if(cmd == CMD_ACCOUNT)           { CmdAccount(); return; }
   if(cmd == CMD_TERMINAL)          { CmdTerminal(); return; }
   if(cmd == CMD_VERSION)           { CmdVersion(); return; }
   if(cmd == CMD_LASTERR)           { CmdLastError(); return; }
   if(cmd == CMD_SYMBOLS_TOTAL)     { CmdSymbolsTotal(); return; }
   if(cmd == CMD_SYMBOLS)           { CmdSymbols(); return; }
   if(cmd == CMD_SYMBOL)            { CmdSymbol(args); return; }
   if(cmd == CMD_TICK)              { CmdTick(args); return; }
   if(cmd == CMD_SELECT)            { CmdSelect(args); return; }
   if(cmd == CMD_RATES_POS)         { CmdRatesPos(args); return; }
   if(cmd == CMD_RATES_RANGE)       { CmdRatesRange(args); return; }
   if(cmd == CMD_ORDERS)            { CmdOrders(); return; }
   if(cmd == CMD_POSITIONS)         { CmdPositions(); return; }
   if(cmd == CMD_ORDER_SEND)        { CmdOrderSend(args); return; }
   if(cmd == CMD_ORDER_CHECK)       { CmdOrderCheck(args); return; }
   if(cmd == CMD_HIST_ORDERS)       { CmdHistOrders(args); return; }
   if(cmd == CMD_HIST_DEALS)        { CmdHistDeals(args); return; }

   SendERR("unknown command: " + cmd);
}

//+------------------------------------------------------------------+
void CmdInit(string args)
{
   if(g_initialized) { SendOK("already"); return; }

   // The EA already runs inside the terminal, so there is nothing to
   // initialize — just verify the terminal is connected to the broker.
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))
   {
      SendERR("terminal not connected to broker");
      return;
   }

   g_initialized = true;
   SendOK("initialized");
}

//+------------------------------------------------------------------+
void CmdShutdown()
{
   g_initialized = false;
   SendOK("shutdown");
}

//+------------------------------------------------------------------+
void CmdAccount()
{
   if(!EnsureInit()) return;
   double bal   = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq    = AccountInfoDouble(ACCOUNT_EQUITY);
   double margin= AccountInfoDouble(ACCOUNT_MARGIN);
   double freeM = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double prof  = AccountInfoDouble(ACCOUNT_PROFIT);
   double credit= AccountInfoDouble(ACCOUNT_CREDIT);
   double mlevel= AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
    long   lev   = AccountInfoInteger(ACCOUNT_LEVERAGE);
   long   login = AccountInfoInteger(ACCOUNT_LOGIN);
   string name  = AccountInfoString(ACCOUNT_NAME);
   string cur   = AccountInfoString(ACCOUNT_CURRENCY);
   string srv   = AccountInfoString(ACCOUNT_SERVER);
    int    mode  = (int)AccountInfoInteger(ACCOUNT_TRADE_MODE);

   string s = "login=" + IntegerToString(login)
            + "|name=" + name
            + "|balance=" + DoubleToString(bal, 2)
            + "|equity=" + DoubleToString(eq, 2)
            + "|margin=" + DoubleToString(margin, 2)
            + "|margin_free=" + DoubleToString(freeM, 2)
            + "|profit=" + DoubleToString(prof, 2)
            + "|credit=" + DoubleToString(credit, 2)
            + "|margin_level=" + DoubleToString(mlevel, 2)
             + "|leverage=" + IntegerToString(lev)
            + "|currency=" + cur
            + "|server=" + srv
            + "|trade_mode=" + IntegerToString(mode);
   SendOK(s);
}

//+------------------------------------------------------------------+
void CmdTerminal()
{
   if(!EnsureInit()) return;
   bool connected = (bool)TerminalInfoInteger(TERMINAL_CONNECTED);
   bool tradeAllowed = (bool)TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   string name = TerminalInfoString(TERMINAL_NAME);
   string path = TerminalInfoString(TERMINAL_DATA_PATH);
   string comm = TerminalInfoString(TERMINAL_COMPANY);
   int build = (int)TerminalInfoInteger(TERMINAL_BUILD);

   string s = "connected=" + (connected ? "1" : "0")
            + "|trade_allowed=" + (tradeAllowed ? "1" : "0")
            + "|name=" + name
            + "|path=" + path
            + "|company=" + comm
            + "|build=" + IntegerToString(build);
   SendOK(s);
}

//+------------------------------------------------------------------+
void CmdVersion()
{
   string s = IntegerToString(TerminalInfoInteger(TERMINAL_BUILD));
   SendOK(s);
}

//+------------------------------------------------------------------+
void CmdLastError()
{
   int err = GetLastError();
   SendOK(IntegerToString(err));
}

//+------------------------------------------------------------------+
void CmdSymbolsTotal()
{
   int total = SymbolsTotal(true);
   SendOK(IntegerToString(total));
}

//+------------------------------------------------------------------+
void CmdSymbols()
{
   int total = SymbolsTotal(true);
   if(total > 500) total = 500; // cap for safety
   for(int i = 0; i < total; i++)
   {
      string sym = SymbolName(i, true);
      if(StringLen(sym) == 0) continue;
      SendResponse("SYM " + sym);
   }
   SendResponse("END");
}

//+------------------------------------------------------------------+
void CmdSymbol(string sym)
{
   if(!EnsureInit()) return;
   if(StringLen(sym) == 0) { SendERR("no symbol"); return; }

   double bid = SymbolInfoDouble(sym, SYMBOL_BID);
   double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   double tickVal = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double contractSize = SymbolInfoDouble(sym, SYMBOL_TRADE_CONTRACT_SIZE);
   int tradeMode = (int)SymbolInfoInteger(sym, SYMBOL_TRADE_MODE);
    int spread = (int)SymbolInfoInteger(sym, SYMBOL_SPREAD);
    string curBase = SymbolInfoString(sym, SYMBOL_CURRENCY_BASE);
    string curProfit = SymbolInfoString(sym, SYMBOL_CURRENCY_PROFIT);
    string curMargin = SymbolInfoString(sym, SYMBOL_CURRENCY_MARGIN);
    string desc = SymbolInfoString(sym, SYMBOL_DESCRIPTION);
    double volMin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
    double volMax = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
    double volStep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
    int stopsLevel = (int)SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);
    int fillMode = (int)SymbolInfoInteger(sym, SYMBOL_FILLING_MODE);

    string s = "symbol=" + sym
             + "|bid=" + DoubleToString(bid, digits)
             + "|ask=" + DoubleToString(ask, digits)
             + "|point=" + DoubleToString(point, digits + 2)
             + "|digits=" + IntegerToString(digits)
             + "|tick_value=" + DoubleToString(tickVal, 6)
             + "|trade_tick_value=" + DoubleToString(tickVal, 6)
             + "|tick_size=" + DoubleToString(tickSize, 8)
             + "|trade_tick_size=" + DoubleToString(tickSize, 8)
             + "|volume_min=" + DoubleToString(volMin, 4)
             + "|volume_max=" + DoubleToString(volMax, 4)
             + "|volume_step=" + DoubleToString(volStep, 4)
             + "|trade_stops_level=" + IntegerToString(stopsLevel)
             + "|filling_mode=" + IntegerToString(fillMode)
             + "|contract_size=" + DoubleToString(contractSize, 2)
             + "|trade_mode=" + IntegerToString(tradeMode)
             + "|spread=" + IntegerToString(spread)
            + "|currency_base=" + curBase
            + "|currency_profit=" + curProfit
            + "|currency_margin=" + curMargin
            + "|description=" + desc;
   SendOK(s);
}

//+------------------------------------------------------------------+
void CmdTick(string sym)
{
   if(!EnsureInit()) return;
   if(StringLen(sym) == 0) { SendERR("no symbol"); return; }

   MqlTick tick;
   if(!SymbolInfoTick(sym, tick))
   {
      SendERR("tick failed: " + IntegerToString(GetLastError()));
      return;
   }

   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   string s = "symbol=" + sym
            + "|bid=" + DoubleToString(tick.bid, digits)
            + "|ask=" + DoubleToString(tick.ask, digits)
            + "|last=" + DoubleToString(tick.last, digits)
            + "|time=" + IntegerToString(tick.time)
            + "|volume=" + IntegerToString(tick.volume_real > 0 ? (int)tick.volume_real : tick.volume);
   SendOK(s);
}

//+------------------------------------------------------------------+
void CmdSelect(string args)
{
   if(!EnsureInit()) return;
   string parts[];
   int n = StringSplit(args, ' ', parts);
   if(n < 2) { SendERR("usage: SELECT symbol enable"); return; }
   string sym = parts[0];
   bool enable = (StringToInteger(parts[1]) != 0);
   bool ok = SymbolSelect(sym, enable);
   if(ok) SendOK("selected");
   else SendERR("select failed: " + IntegerToString(GetLastError()));
}

//+------------------------------------------------------------------+
void CmdRatesPos(string args)
{
   if(!EnsureInit()) return;
   string parts[];
   int n = StringSplit(args, ' ', parts);
   if(n < 4) { SendERR("usage: RATES_POS symbol tf pos count"); return; }

    string sym = parts[0];
    int tf = (int)StringToInteger(parts[1]);
    int pos = (int)StringToInteger(parts[2]);
    int count = (int)StringToInteger(parts[3]);

   MqlRates rates[];
   int copied = CopyRates(sym, (ENUM_TIMEFRAMES)tf, pos, count, rates);
   if(copied <= 0)
   {
      SendERR("copy_rates failed: " + IntegerToString(GetLastError()));
      return;
   }

   SendResponse("COUNT " + IntegerToString(copied));
   for(int i = 0; i < copied; i++)
   {
      string line = "BAR t=" + IntegerToString(rates[i].time)
                  + "|o=" + DoubleToString(rates[i].open, 8)
                  + "|h=" + DoubleToString(rates[i].high, 8)
                  + "|l=" + DoubleToString(rates[i].low, 8)
                  + "|c=" + DoubleToString(rates[i].close, 8)
                  + "|v=" + IntegerToString(rates[i].tick_volume)
                  + "|s=" + IntegerToString(rates[i].spread)
                  + "|rv=" + IntegerToString((int)rates[i].real_volume);
      SendResponse(line);
   }
   SendResponse("END");
}

//+------------------------------------------------------------------+
void CmdRatesRange(string args)
{
   if(!EnsureInit()) return;
   string parts[];
   int n = StringSplit(args, ' ', parts);
   if(n < 4) { SendERR("usage: RATES_RANGE symbol tf start end"); return; }

    string sym = parts[0];
    int tf = (int)StringToInteger(parts[1]);
    datetime start = (datetime)StringToInteger(parts[2]);
    datetime end = (datetime)StringToInteger(parts[3]);

   MqlRates rates[];
   int copied = CopyRates(sym, (ENUM_TIMEFRAMES)tf, start, end, rates);
   if(copied <= 0)
   {
      SendERR("copy_rates failed: " + IntegerToString(GetLastError()));
      return;
   }

   SendResponse("COUNT " + IntegerToString(copied));
   for(int i = 0; i < copied; i++)
   {
      string line = "BAR t=" + IntegerToString(rates[i].time)
                  + "|o=" + DoubleToString(rates[i].open, 8)
                  + "|h=" + DoubleToString(rates[i].high, 8)
                  + "|l=" + DoubleToString(rates[i].low, 8)
                  + "|c=" + DoubleToString(rates[i].close, 8)
                  + "|v=" + IntegerToString(rates[i].tick_volume)
                  + "|s=" + IntegerToString(rates[i].spread)
                  + "|rv=" + IntegerToString((int)rates[i].real_volume);
      SendResponse(line);
   }
   SendResponse("END");
}

//+------------------------------------------------------------------+
void CmdOrders()
{
   if(!EnsureInit()) return;
   int total = OrdersTotal();
   SendResponse("COUNT " + IntegerToString(total));
    for(int i = 0; i < total; i++)
    {
       ulong ticket = OrderGetTicket(i);
       if(ticket == 0) continue;
       string line = "ORD tkt=" + IntegerToString(ticket)
                   + "|sym=" + OrderGetString(ORDER_SYMBOL)
                   + "|type=" + IntegerToString((int)OrderGetInteger(ORDER_TYPE))
                   + "|vol=" + DoubleToString(OrderGetDouble(ORDER_VOLUME_CURRENT), 4)
                   + "|price=" + DoubleToString(OrderGetDouble(ORDER_PRICE_OPEN), 8)
                   + "|sl=" + DoubleToString(OrderGetDouble(ORDER_SL), 8)
                   + "|tp=" + DoubleToString(OrderGetDouble(ORDER_TP), 8)
                   + "|magic=" + IntegerToString((int)OrderGetInteger(ORDER_MAGIC))
                   + "|comment=" + OrderGetString(ORDER_COMMENT);
       SendResponse(line);
    }
   SendResponse("END");
}

//+------------------------------------------------------------------+
void CmdPositions()
{
   if(!EnsureInit()) return;
   int total = PositionsTotal();
   SendResponse("COUNT " + IntegerToString(total));
    for(int i = 0; i < total; i++)
    {
       ulong ticket = PositionGetTicket(i);
       if(ticket == 0) continue;
       if(!PositionSelectByTicket(ticket)) continue;
      string sym = PositionGetString(POSITION_SYMBOL);
      long type = PositionGetInteger(POSITION_TYPE);
      double vol = PositionGetDouble(POSITION_VOLUME);
      double price = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double profit = PositionGetDouble(POSITION_PROFIT);
      double swap = PositionGetDouble(POSITION_SWAP);
      long magic = PositionGetInteger(POSITION_MAGIC);
      string comment = PositionGetString(POSITION_COMMENT);

      string line = "POS tkt=" + IntegerToString(ticket)
                  + "|sym=" + sym
                  + "|type=" + IntegerToString((int)type)
                  + "|vol=" + DoubleToString(vol, 4)
                  + "|price=" + DoubleToString(price, 8)
                  + "|sl=" + DoubleToString(sl, 8)
                  + "|tp=" + DoubleToString(tp, 8)
                  + "|profit=" + DoubleToString(profit, 2)
                  + "|swap=" + DoubleToString(swap, 2)
                  + "|magic=" + IntegerToString((int)magic)
                  + "|comment=" + comment;
      SendResponse(line);
   }
   SendResponse("END");
}

//+------------------------------------------------------------------+
void CmdOrderSend(string args)
{
   if(!EnsureInit()) return;

   // parse key=value pairs separated by |
   string pairs[];
   int n = StringSplit(args, '|', pairs);

   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   for(int i = 0; i < n; i++)
   {
      string kv[];
      if(StringSplit(pairs[i], '=', kv) < 2) continue;
      string key = kv[0];
      string val = kv[1];
      StringToUpper(key);

      if(key == "ACTION")      request.action = (ENUM_TRADE_REQUEST_ACTIONS)StringToInteger(val);
      else if(key == "SYMBOL") request.symbol = val;
      else if(key == "VOLUME") request.volume = StringToDouble(val);
      else if(key == "TYPE")   request.type = (ENUM_ORDER_TYPE)StringToInteger(val);
      else if(key == "PRICE")  request.price = StringToDouble(val);
      else if(key == "SL")     request.sl = StringToDouble(val);
      else if(key == "TP")     request.tp = StringToDouble(val);
      else if(key == "MAGIC")  request.magic = StringToInteger(val);
      else if(key == "COMMENT") request.comment = val;
      else if(key == "TYPE_TIME") request.type_time = (ENUM_ORDER_TYPE_TIME)StringToInteger(val);
      else if(key == "TYPE_FILLING") request.type_filling = (ENUM_ORDER_TYPE_FILLING)StringToInteger(val);
      else if(key == "POSITION") request.position = StringToInteger(val);
   }

   bool ok = OrderSend(request, result);
   if(!ok)
   {
      SendERR("order_send failed: retcode=" + IntegerToString(result.retcode)
            + "|code=" + IntegerToString(GetLastError())
            + "|comment=" + result.comment);
      return;
   }

   string s = "retcode=" + IntegerToString(result.retcode)
            + "|order=" + IntegerToString(result.order)
            + "|deal=" + IntegerToString(result.deal)
            + "|volume=" + DoubleToString(result.volume, 4)
            + "|price=" + DoubleToString(result.price, 8)
            + "|comment=" + result.comment;
   SendOK(s);
}

//+------------------------------------------------------------------+
void CmdOrderCheck(string args)
{
   if(!EnsureInit()) return;

   string pairs[];
   int n = StringSplit(args, '|', pairs);

   MqlTradeRequest request = {};
   MqlTradeCheckResult check = {};

   for(int i = 0; i < n; i++)
   {
      string kv[];
      if(StringSplit(pairs[i], '=', kv) < 2) continue;
      string key = kv[0];
      string val = kv[1];
      StringToUpper(key);

      if(key == "ACTION")      request.action = (ENUM_TRADE_REQUEST_ACTIONS)StringToInteger(val);
      else if(key == "SYMBOL") request.symbol = val;
      else if(key == "VOLUME") request.volume = StringToDouble(val);
      else if(key == "TYPE")   request.type = (ENUM_ORDER_TYPE)StringToInteger(val);
      else if(key == "PRICE")  request.price = StringToDouble(val);
      else if(key == "SL")     request.sl = StringToDouble(val);
      else if(key == "TP")     request.tp = StringToDouble(val);
      else if(key == "MAGIC")  request.magic = StringToInteger(val);
      else if(key == "COMMENT") request.comment = val;
      else if(key == "TYPE_TIME") request.type_time = (ENUM_ORDER_TYPE_TIME)StringToInteger(val);
      else if(key == "TYPE_FILLING") request.type_filling = (ENUM_ORDER_TYPE_FILLING)StringToInteger(val);
      else if(key == "POSITION") request.position = StringToInteger(val);
   }

   bool ok = OrderCheck(request, check);
   string s = "retcode=" + IntegerToString(check.retcode)
            + "|balance=" + DoubleToString(check.balance, 2)
            + "|equity=" + DoubleToString(check.equity, 2)
            + "|profit=" + DoubleToString(check.profit, 2)
            + "|margin=" + DoubleToString(check.margin, 2)
            + "|comment=" + check.comment;
   if(ok) SendOK(s);
   else SendERR("check failed: " + s);
}

//+------------------------------------------------------------------+
void CmdHistOrders(string args)
{
   if(!EnsureInit()) return;
   string parts[];
   int n = StringSplit(args, ' ', parts);
   if(n < 2) { SendERR("usage: HIST_ORDERS start end"); return; }

   datetime start = (datetime)StringToInteger(parts[0]);
   datetime end = (datetime)StringToInteger(parts[1]);

   HistorySelect(start, end);
   int total = HistoryOrdersTotal();
   SendResponse("COUNT " + IntegerToString(total));
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryOrderGetInteger(i, ORDER_TICKET);
      string sym = HistoryOrderGetString(i, ORDER_SYMBOL);
      string line = "HORD tkt=" + IntegerToString(ticket)
                  + "|sym=" + sym
                  + "|type=" + IntegerToString((int)HistoryOrderGetInteger(i, ORDER_TYPE))
                  + "|state=" + IntegerToString((int)HistoryOrderGetInteger(i, ORDER_STATE))
                  + "|vol_init=" + DoubleToString(HistoryOrderGetDouble(i, ORDER_VOLUME_INITIAL), 4)
                  + "|vol_cur=" + DoubleToString(HistoryOrderGetDouble(i, ORDER_VOLUME_CURRENT), 4)
                  + "|price=" + DoubleToString(HistoryOrderGetDouble(i, ORDER_PRICE_OPEN), 8)
                  + "|sl=" + DoubleToString(HistoryOrderGetDouble(i, ORDER_SL), 8)
                  + "|tp=" + DoubleToString(HistoryOrderGetDouble(i, ORDER_TP), 8)
                  + "|magic=" + IntegerToString((int)HistoryOrderGetInteger(i, ORDER_MAGIC))
                  + "|time_setup=" + IntegerToString((datetime)HistoryOrderGetInteger(i, ORDER_TIME_SETUP))
                  + "|time_done=" + IntegerToString((datetime)HistoryOrderGetInteger(i, ORDER_TIME_DONE));
      SendResponse(line);
   }
   SendResponse("END");
}

//+------------------------------------------------------------------+
void CmdHistDeals(string args)
{
   if(!EnsureInit()) return;
   string parts[];
   int n = StringSplit(args, ' ', parts);
   if(n < 2) { SendERR("usage: HIST_DEALS start end"); return; }

   datetime start = (datetime)StringToInteger(parts[0]);
   datetime end = (datetime)StringToInteger(parts[1]);

   HistorySelect(start, end);
   int total = HistoryDealsTotal();
   SendResponse("COUNT " + IntegerToString(total));
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetInteger(i, DEAL_TICKET);
      ulong order = HistoryDealGetInteger(i, DEAL_ORDER);
      string sym = HistoryDealGetString(i, DEAL_SYMBOL);
      string line = "HDEAL tkt=" + IntegerToString(ticket)
                  + "|order=" + IntegerToString(order)
                  + "|sym=" + sym
                  + "|type=" + IntegerToString((int)HistoryDealGetInteger(i, DEAL_TYPE))
                  + "|entry=" + IntegerToString((int)HistoryDealGetInteger(i, DEAL_ENTRY))
                  + "|volume=" + DoubleToString(HistoryDealGetDouble(i, DEAL_VOLUME), 4)
                  + "|price=" + DoubleToString(HistoryDealGetDouble(i, DEAL_PRICE), 8)
                  + "|profit=" + DoubleToString(HistoryDealGetDouble(i, DEAL_PROFIT), 2)
                  + "|swap=" + DoubleToString(HistoryDealGetDouble(i, DEAL_SWAP), 2)
                  + "|commission=" + DoubleToString(HistoryDealGetDouble(i, DEAL_COMMISSION), 2)
                  + "|magic=" + IntegerToString((int)HistoryDealGetInteger(i, DEAL_MAGIC))
                  + "|time=" + IntegerToString((datetime)HistoryDealGetInteger(i, DEAL_TIME));
      SendResponse(line);
   }
   SendResponse("END");
}

//+------------------------------------------------------------------+
bool EnsureInit()
{
   if(g_initialized) return true;
   SendERR("not initialized");
   return false;
}

//+------------------------------------------------------------------+
bool TerminalInitialize(string login, string password, string server)
{
   // MT5 terminal is already running — just verify connection
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))
   {
      // try to connect
      if(StringLen(login) > 0 && StringLen(password) > 0 && StringLen(server) > 0)
      {
         // can't auto-login from EA, but terminal should already be logged in
      }
   }
   return (bool)TerminalInfoInteger(TERMINAL_CONNECTED);
}
//+------------------------------------------------------------------+
