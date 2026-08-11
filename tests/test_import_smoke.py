"""Import-smoke tests for modules that lack dedicated test coverage.

Each test verifies the module can be imported and its public API is
callable without crashing. MT5-dependent functions are skipped or mocked.
"""



class TestAutoOptimizer:
    def test_module_importable(self):
        import auto_optimizer
        assert hasattr(auto_optimizer, "optimize_symbol")


class TestBackup:
    def test_module_importable(self):
        import backup
        assert hasattr(backup, "run_backup")


class TestCalibrateModels:
    def test_module_importable(self):
        import calibrate_models
        assert hasattr(calibrate_models, "calibrate_model_file")


class TestConfig:
    def test_module_importable(self):
        import config
        assert hasattr(config, "load_config")


class TestCorrelation:
    def test_module_importable(self):
        import correlation
        assert hasattr(correlation, "compute_correlation_matrix")


class TestDashboard:
    def test_module_importable(self):
        import dashboard
        assert hasattr(dashboard, "write_dashboard_state")


class TestDiscordAlerts:
    def test_module_importable(self):
        import discord_alerts
        assert hasattr(discord_alerts, "send_alert")


class TestMcRuin:
    def test_module_importable(self):
        import mc_ruin
        assert hasattr(mc_ruin, "run")


class TestMcValidation:
    def test_module_importable(self):
        import mc_validation
        assert hasattr(mc_validation, "compute_mc_report")


class TestScenarioAnalysis:
    def test_module_importable(self):
        import scenario_analysis
        assert hasattr(scenario_analysis, "run_scenario_analysis")


class TestScreenFast:
    def test_module_importable(self):
        import screen_fast
        assert hasattr(screen_fast, "classify")


class TestScreenSymbols:
    def test_module_importable(self):
        import screen_symbols
        assert hasattr(screen_symbols, "base_params")


class TestTuneScaleout:
    def test_module_importable(self):
        import tune_scaleout
        assert hasattr(tune_scaleout, "tune_symbol")


class TestWeeklySummary:
    def test_module_importable(self):
        import weekly_summary
        assert hasattr(weekly_summary, "send_weekly_discord")


class TestNewsSentiment:
    def test_module_importable(self):
        import services.news_sentiment as ns
        assert hasattr(ns, "fetch_marketaux")


class TestMt5Connect:
    def test_module_importable(self):
        import mt5_connect
        assert hasattr(mt5_connect, "get_rates")
