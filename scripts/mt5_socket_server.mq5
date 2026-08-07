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

input int    SocketPort = 9000;
input int    PollIntervalMs = 50;
input int    MaxBufferSize = 65536;

//--- globals
int    g_serverSocket = INVALID_HANDLE;
int    g_clientSocket = INVALID_HANDLE;
bool   g_initialized  = false;
bool   g_clientConnected = false;
string g_readBuffer   = "";
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
   g_serverSocket = SocketCreate();
   if(g_serverSocket == INVALID_HANDLE)
   {
      Print("[MT5Socket] SocketCreate failed: ", GetLastError());
      return INIT_FAILED;
   }

   if(!SocketBind(g_serverSocket, SocketPort, "127.0.0.1"))
   {
      Print("[MT5Socket] SocketBind failed: ", GetLastError());
      SocketClose(g_serverSocket);
      return INIT_FAILED;
   }

   if(!SocketListen(g_serverSocket, 1))
   {
      Print("[MT5Socket] SocketListen failed: ", GetLastError());
      SocketClose(g_serverSocket);
      return INIT_FAILED;
   }

   EventSetTimer(PollIntervalMs / 1000);
   Print("[MT5Socket] Listening on 127.0.0.1:", SocketPort);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   if(g_clientSocket != INVALID_HANDLE) SocketClose(g_clientSocket);
   if(g_serverSocket != INVALID_HANDLE) SocketClose(g_serverSocket);
   Print("[MT5Socket] Stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
void OnTimer()
{
   //--- accept new connection if none
   if(!g_clientConnected)
   {
      int accepted = SocketAccept(g_serverSocket, 1);
      if(accepted != INVALID_HANDLE)
      {
         g_clientSocket = accepted;
         g_clientConnected = true;
         g_readBuffer = "";
         Print("[MT5Socket] Client connected");
      }
      return;
   }

   //--- read available data
   uchar buf[];
   int bytes = SocketRead(g_clientSocket, buf, MaxBufferSize, 0);
   if(bytes < 0)
   {
      int err = GetLastError();
      if(err != 40002) // WSAEWOULDBLOCK
      {
         Print("[MT5Socket] Client disconnected: ", err);
         SocketClose(g_clientSocket);
         g_clientSocket = INVALID_HANDLE;
         g_clientConnected = false;
      }
      return;
   }
   if(bytes == 0) return;

   //--- convert to string and append to buffer
   string chunk = CharArrayToString(buf, 0, bytes);
   g_readBuffer += chunk;

   //--- process complete lines
   int pos;
   while((pos = StringFind(g_readBuffer, "\n")) >= 0)
   {
      StringReplace(g_readBuffer, "\r", "");
      StringReplace(g_readBuffer, "\n", "");
      // re-find after cleanup
      pos = StringFind(g_readBuffer, "\n");
      if(pos < 0) break;
      line = StringSubstr(g_readBuffer, 0, pos);
      g_readBuffer = StringSubstr(g_readBuffer, pos + 1);
      ProcessCommand(line);
   }
}

//+------------------------------------------------------------------+
void SendResponse(string msg)
{
   if(!g_clientConnected || g_clientSocket == INVALID_HANDLE) return;
   msg += "\n";
   uchar buf[];
   StringToCharArray(msg, buf);
   SocketSend(g_clientSocket, buf, StringLen(msg));
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

   string parts[];
   int n = StringSplit(args, ' ', parts);

   bool ok = false;
   if(n >= 3)
      ok = TerminalInitialize(IntegerToString(parts[0]), parts[1], parts[2]);
   else
      ok = TerminalInitialize("", "", "");

   if(ok)
   {
      g_initialized = true;
      SendOK("initialized");
   }
   else
   {
      int err = GetLastError();
      SendERR("init failed: " + IntegerToString(err));
   }
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
   double lev   = AccountInfoDouble(ACCOUNT_LEVERAGE);
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
            + "|leverage=" + IntegerToString((int)lev)
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
   double spread = SymbolInfoDouble(sym, SYMBOL_SPREAD);
   string curBase = SymbolInfoString(sym, SYMBOL_CURRENCY_BASE);
   string curProfit = SymbolInfoString(sym, SYMBOL_CURRENCY_PROFIT);
   string curMargin = SymbolInfoString(sym, SYMBOL_CURRENCY_MARGIN);
   string desc = SymbolInfoString(sym, SYMBOL_DESCRIPTION);

   string s = "symbol=" + sym
            + "|bid=" + DoubleToString(bid, digits)
            + "|ask=" + DoubleToString(ask, digits)
            + "|point=" + DoubleToString(point, digits + 2)
            + "|digits=" + IntegerToString(digits)
            + "|tick_value=" + DoubleToString(tickVal, 6)
            + "|tick_size=" + DoubleToString(tickSize, 8)
            + "|contract_size=" + DoubleToString(contractSize, 2)
            + "|trade_mode=" + IntegerToString(tradeMode)
            + "|spread=" + DoubleToString(spread, 1)
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
   int tf = StringToInteger(parts[1]);
   int pos = StringToInteger(parts[2]);
   int count = StringToInteger(parts[3]);

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
   int tf = StringToInteger(parts[1]);
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
      ulong ticket = OrderGetInteger(ORDER_TICKET);
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
      if(!OrderSelectByIndex(i)) break;
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
      if(!PositionSelectByIndex(i)) continue;
      ulong ticket = PositionGetInteger(POSITION_TICKET);
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
