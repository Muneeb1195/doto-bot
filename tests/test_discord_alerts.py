"""Tests for discord_alerts.py — send_alert payload shape, trade_open/close embeds."""

from unittest.mock import MagicMock


class TestSendAlert:
    def test_sends_post_with_embed(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            resp = MagicMock()
            resp.status_code = 200
            return resp

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import send_alert
        send_alert("https://discord.com/api/webhooks/valid", "INFO", "Test Title",
                   [("Key1", "Val1"), ("Key2", "Val2")])
        assert posted[0] is not None
        embeds = posted[0]["embeds"]
        assert len(embeds) == 1
        embed = embeds[0]
        assert embed["title"] == "Test Title"
        assert len(embed["fields"]) == 2
        assert embed["fields"][0]["name"] == "Key1"
        assert embed["fields"][0]["value"] == "Val1"

    def test_skips_when_url_is_placeholder(self, monkeypatch):
        called = [False]

        def fake_post(*a, **kw):
            called[0] = True
            resp = MagicMock()
            resp.status_code = 200
            return resp

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import send_alert
        send_alert("https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE", "INFO", "T", [])
        assert not called[0]

    def test_skips_when_url_empty(self, monkeypatch):
        called = [False]

        def fake_post(*a, **kw):
            called[0] = True
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import send_alert
        send_alert("", "INFO", "T", [])
        assert not called[0]

    def test_retries_on_429(self, monkeypatch, caplog):
        attempts = [0]

        def fake_post(url, json, timeout):
            attempts[0] += 1
            resp = MagicMock()
            resp.status_code = 200 if attempts[0] > 1 else 429
            return resp

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import send_alert
        send_alert("https://discord.com/api/webhooks/valid", "INFO", "Retry", [])
        assert attempts[0] == 2

    def test_uses_custom_color(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import send_alert
        send_alert("https://discord.com/api/webhooks/valid", "BUY", "Buy Alert",
                   [("Sym", "X")], color=0x00FF00)
        assert posted[0]["embeds"][0]["color"] == 0x00FF00

    def test_defaults_to_event_color(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import send_alert
        send_alert("https://discord.com/api/webhooks/valid", "ERROR", "Err", [("M", "msg")])
        assert posted[0]["embeds"][0]["color"] == 0xE74C3C

    def test_logs_error_on_non_200(self, monkeypatch, caplog):
        def fake_post(url, json, timeout):
            resp = MagicMock()
            resp.status_code = 400
            resp.text = "Bad Request"
            return resp

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import send_alert
        send_alert("https://discord.com/api/webhooks/valid", "INFO", "Test", [("K", "V")])
        assert "400" in caplog.text

    def test_inline_fields(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import send_alert
        send_alert("https://discord.com/api/webhooks/valid", "INFO", "Test",
                   [("A", "1"), ("B", "2")])
        for field in posted[0]["embeds"][0]["fields"]:
            assert field["inline"] is True


class TestTradeOpen:
    def test_sends_embed(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import trade_open
        trade_open("https://discord.com/api/webhooks/valid", "XAU500.raw", "buy",
                   0.1, 1900.0, 1880.0, 1920.0, 15.0, "strong_trend")
        embed = posted[0]["embeds"][0]
        assert "XAU500" in embed["title"]
        assert embed["color"] == 0x00FF00
        field_names = [f["name"] for f in embed["fields"]]
        assert "Volume" in field_names
        assert "Entry" in field_names
        assert "SL" in field_names
        assert "TP" in field_names
        assert "ATR" in field_names
        assert "Regime" in field_names


class TestTradeClose:
    def test_sends_embed(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import trade_close
        trade_close("https://discord.com/api/webhooks/valid", "BTCUSD.raw", "sell",
                    0.05, 64000.0, 63800.0, 200.0, 20.0, "TP hit")
        embed = posted[0]["embeds"][0]
        assert "BTCUSD" in embed["title"]
        assert embed["color"] == 0xFF0000
        field_names = [f["name"] for f in embed["fields"]]
        assert "P&L" in field_names
        assert "Pips" in field_names
        assert "Reason" in field_names
        assert "Exit" in field_names


class TestTradePartial:
    def test_sends_embed(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import trade_partial
        trade_partial("https://discord.com/api/webhooks/valid", "XAU500.raw", "buy", 0.05, 1910.0, 50.0)
        embed = posted[0]["embeds"][0]
        assert "PARTIAL" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "Closed" in field_names
        assert "Price" in field_names
        assert "P&L" in field_names


class TestDailySummary:
    def test_sends_embed(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import daily_summary
        daily_summary("https://discord.com/api/webhooks/valid", 500000.0, 501000.0, 1000.0, 3, 66.7, 5)
        embed = posted[0]["embeds"][0]
        assert "Daily Summary" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "Balance" in field_names
        assert "Equity" in field_names
        assert "Daily P&L" in field_names
        assert "Win Rate" in field_names


class TestBotStart:
    def test_sends_embed(self, monkeypatch):
        posted = [None]

        def fake_post(url, json, timeout):
            posted[0] = json
            return MagicMock(status_code=200)

        monkeypatch.setattr("discord_alerts.requests.post", fake_post)
        from discord_alerts import bot_start
        bot_start("https://discord.com/api/webhooks/valid", ["XAU500.raw", "BTCUSD.raw"], 500000.0)
        embed = posted[0]["embeds"][0]
        assert "Bot Started" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "Symbols" in field_names
        assert "Balance" in field_names
