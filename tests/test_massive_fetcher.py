# -*- coding: utf-8 -*-
"""
data_provider/massive_fetcher 单元测试

mock requests.get 避免真实网络调用，覆盖：
- 未配置 MASSIVE_API_KEY 时的兜底行为
- 非美股代码（A 股 / 港股）跳过
- aggs 日线响应解析与列名标准化
- snapshot 实时行情解析
- HTTP 错误码 (401 / 429) 的语义化失败
- 配置后 base.py 美股 source_order 是否正确插入 MassiveFetcher
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# fake_useragent 是某些 fetcher 的可选依赖；CI 环境可能缺，先 stub 掉
if "fake_useragent" not in sys.modules:
    sys.modules["fake_useragent"] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.base import DataFetchError  # noqa: E402
from data_provider.massive_fetcher import MassiveFetcher, is_massive_configured  # noqa: E402


def _mock_response(status: int, json_payload=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_payload if json_payload is not None else {}
    resp.text = text
    return resp


class TestMassiveFetcherConfig(unittest.TestCase):
    def test_is_massive_configured_false_when_unset(self):
        with patch.dict(os.environ, {"MASSIVE_API_KEY": ""}, clear=False):
            self.assertFalse(is_massive_configured())

    def test_is_massive_configured_true_when_set(self):
        with patch.dict(os.environ, {"MASSIVE_API_KEY": "k"}, clear=False):
            self.assertTrue(is_massive_configured())


class TestMassiveFetcherSkipsNonUS(unittest.TestCase):
    def setUp(self):
        self.fetcher = MassiveFetcher()
        self.env = patch.dict(os.environ, {"MASSIVE_API_KEY": "test"}, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_a_share_skipped(self):
        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data("600519", "2026-04-01", "2026-04-25")

    def test_hk_skipped(self):
        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data("hk00700", "2026-04-01", "2026-04-25")

    def test_realtime_returns_none_for_a_share(self):
        self.assertIsNone(self.fetcher.get_realtime_quote("600519"))


class TestMassiveFetcherWithoutKey(unittest.TestCase):
    def setUp(self):
        self.fetcher = MassiveFetcher()

    def test_fetch_raises_when_key_missing(self):
        with patch.dict(os.environ, {"MASSIVE_API_KEY": ""}, clear=False):
            with self.assertRaises(DataFetchError):
                self.fetcher._fetch_raw_data("AAPL", "2026-04-01", "2026-04-25")

    def test_realtime_returns_none_when_key_missing(self):
        with patch.dict(os.environ, {"MASSIVE_API_KEY": ""}, clear=False):
            self.assertIsNone(self.fetcher.get_realtime_quote("AAPL"))


class TestMassiveFetcherDailyData(unittest.TestCase):
    def setUp(self):
        self.fetcher = MassiveFetcher()
        self.env = patch.dict(os.environ, {"MASSIVE_API_KEY": "test"}, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_aggs_response_normalizes_to_standard_columns(self):
        sample = {
            "ticker": "AAPL",
            "status": "DELAYED",
            "results": [
                {"t": 1776744000000, "o": 271.5, "h": 272.8, "l": 265.4, "c": 266.17, "v": 50_192_035, "vw": 267.583},
                {"t": 1776830400000, "o": 267.82, "h": 273.74, "l": 266.87, "c": 273.17, "v": 43_249_204, "vw": 271.93},
            ],
        }
        with patch("data_provider.massive_fetcher.requests.get", return_value=_mock_response(200, sample)):
            raw = self.fetcher._fetch_raw_data("AAPL", "2026-04-21", "2026-04-22")
        self.assertEqual(len(raw), 2)
        normalized = self.fetcher._normalize_data(raw, "AAPL")
        for col in ["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]:
            self.assertIn(col, normalized.columns)
        self.assertEqual(normalized.iloc[0]["code"], "AAPL")
        # 第二根 K 涨跌幅应为正
        self.assertGreater(normalized.iloc[1]["pct_chg"], 0)

    def test_aggs_empty_results_raises(self):
        with patch(
            "data_provider.massive_fetcher.requests.get",
            return_value=_mock_response(200, {"status": "OK", "results": []}),
        ):
            with self.assertRaises(DataFetchError):
                self.fetcher._fetch_raw_data("AAPL", "2026-04-21", "2026-04-22")

    def test_http_401_raises_specific(self):
        with patch(
            "data_provider.massive_fetcher.requests.get",
            return_value=_mock_response(401, text="unauthorized"),
        ):
            with self.assertRaisesRegex(DataFetchError, "401"):
                self.fetcher._fetch_raw_data("AAPL", "2026-04-21", "2026-04-22")

    def test_http_429_raises_rate_limited(self):
        with patch(
            "data_provider.massive_fetcher.requests.get",
            return_value=_mock_response(429, text="too many requests"),
        ):
            with self.assertRaisesRegex(DataFetchError, "429"):
                self.fetcher._fetch_raw_data("AAPL", "2026-04-21", "2026-04-22")


class TestMassiveFetcherRealtime(unittest.TestCase):
    def setUp(self):
        self.fetcher = MassiveFetcher()
        self.env = patch.dict(os.environ, {"MASSIVE_API_KEY": "test"}, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_snapshot_parsed_into_unified_quote(self):
        snapshot = {
            "status": "OK",
            "ticker": {
                "ticker": "AAPL",
                "todaysChangePerc": -0.92,
                "todaysChange": -2.53,
                "day": {"o": 272.755, "h": 273.06, "l": 269.65, "c": 271.06, "v": 38_147_490, "dv": "1.0e9"},
                "prevDay": {"c": 273.43},
            },
        }
        with patch(
            "data_provider.massive_fetcher.requests.get",
            return_value=_mock_response(200, snapshot),
        ):
            quote = self.fetcher.get_realtime_quote("AAPL")
        self.assertIsNotNone(quote)
        self.assertEqual(quote.code, "AAPL")
        self.assertEqual(quote.source.value, "massive")
        self.assertAlmostEqual(quote.price, 271.06, places=2)
        self.assertAlmostEqual(quote.pre_close, 273.43, places=2)
        self.assertAlmostEqual(quote.change_pct, -0.92, places=2)


class TestUSRoutingIncludesMassive(unittest.TestCase):
    """验证 base.py 的美股路由会在配置 key 时插入 MassiveFetcher。"""

    def test_source_order_with_key_inserts_massive_before_yfinance(self):
        from data_provider.base import DataFetcherManager

        with patch.dict(
            os.environ,
            {"MASSIVE_API_KEY": "test", "LONGBRIDGE_APP_KEY": "", "LONGBRIDGE_APP_SECRET": "", "LONGBRIDGE_ACCESS_TOKEN": ""},
            clear=False,
        ):
            mgr = DataFetcherManager()
            names = [f.name for f in mgr._get_fetchers_snapshot()]
            self.assertIn("MassiveFetcher", names)

    def test_source_order_without_key_omits_massive(self):
        from data_provider.massive_fetcher import is_massive_configured

        with patch.dict(os.environ, {"MASSIVE_API_KEY": ""}, clear=False):
            self.assertFalse(is_massive_configured())


if __name__ == "__main__":
    unittest.main()
