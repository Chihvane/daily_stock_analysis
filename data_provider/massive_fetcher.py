# -*- coding: utf-8 -*-
"""
===================================
MassiveFetcher - 美股数据源 (Massive.com REST，Polygon 兼容)
===================================

数据来源：https://api.massive.com/v2 （Polygon-style REST，仅美股）
认证：Authorization: Bearer <MASSIVE_API_KEY>
定位：美股 fallback 数据源；未配置 MASSIVE_API_KEY 时所有方法返回 None，与未启用等价
注意：
  - 仅覆盖美股（路径里写死 locale/us），港股/A 股一律跳过
  - 个人/免费 tier 返回 status="DELAYED"（延迟约 15 分钟），不适合实时盘中
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote, safe_float
from .us_index_mapping import is_us_index_code, is_us_stock_code

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.massive.com"
_DEFAULT_TIMEOUT = 15
_DEFAULT_PRIORITY = 4


def _api_key() -> str:
    return (os.getenv("MASSIVE_API_KEY") or "").strip()


def _base_url() -> str:
    return (os.getenv("MASSIVE_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")


def _timeout_seconds() -> int:
    raw = (os.getenv("MASSIVE_TIMEOUT_SECONDS") or "").strip()
    try:
        return max(1, int(raw)) if raw else _DEFAULT_TIMEOUT
    except ValueError:
        return _DEFAULT_TIMEOUT


def is_massive_configured() -> bool:
    """Returns True when MASSIVE_API_KEY is set; used by routing logic."""
    return bool(_api_key())


class MassiveFetcher(BaseFetcher):
    """Massive.com REST 美股数据源（Polygon 兼容）"""

    name = "MassiveFetcher"
    priority = int(os.getenv("MASSIVE_PRIORITY", str(_DEFAULT_PRIORITY)))

    def __init__(self):
        # 不在构造期校验 key，允许进程启动后再补 .env / Web 设置页
        pass

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _request_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        key = _api_key()
        if not key:
            raise DataFetchError("MASSIVE_API_KEY 未配置")
        url = f"{_base_url()}{path}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {key}"},
            params=params or {},
            timeout=_timeout_seconds(),
        )
        if resp.status_code == 401:
            raise DataFetchError("Massive API key 无效或已过期 (HTTP 401)")
        if resp.status_code == 429:
            raise DataFetchError("Massive API 触发限流 (HTTP 429)")
        if resp.status_code >= 500:
            raise requests.ConnectionError(f"Massive 服务端错误 HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise DataFetchError(f"Massive API HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as e:
            raise DataFetchError(f"Massive 响应非 JSON: {e}") from e

    @staticmethod
    def _is_supported(stock_code: str) -> bool:
        code = (stock_code or "").strip().upper()
        return is_us_stock_code(code) or is_us_index_code(code)

    # ------------------------------------------------------------------
    # BaseFetcher 抽象方法实现
    # ------------------------------------------------------------------
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not _api_key():
            raise DataFetchError("MASSIVE_API_KEY 未配置")
        if not self._is_supported(stock_code):
            raise DataFetchError(f"Massive 仅支持美股，跳过 {stock_code}")

        ticker = stock_code.strip().upper()
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 5000}
        payload = self._request_json(path, params=params)
        results = payload.get("results") or []
        if not results:
            raise DataFetchError(
                f"Massive 未返回 {ticker} 的日线数据 (status={payload.get('status')!r})"
            )
        return pd.DataFrame(results)

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        Polygon 风格列：
            t = 时间戳(ms), o/h/l/c = 开高低收, v = 成交量, vw = 加权均价, n = 笔数
        """
        if df.empty:
            return df
        out = df.copy()
        if "t" in out.columns:
            out["date"] = pd.to_datetime(out["t"], unit="ms", utc=True).dt.tz_convert(
                "America/New_York"
            ).dt.date
        column_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        out = out.rename(columns=column_map)
        if "close" in out.columns:
            out["pct_chg"] = (out["close"].pct_change() * 100).fillna(0).round(2)
        if "vw" in out.columns and "volume" in out.columns:
            out["amount"] = (out["volume"] * out["vw"]).round(2)
        elif "volume" in out.columns and "close" in out.columns:
            out["amount"] = (out["volume"] * out["close"]).round(2)
        else:
            out["amount"] = 0
        out["code"] = stock_code.strip().upper()

        keep_cols = ["code"] + STANDARD_COLUMNS
        existing = [c for c in keep_cols if c in out.columns]
        return out[existing]

    # ------------------------------------------------------------------
    # 实时行情
    # ------------------------------------------------------------------
    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        if not _api_key():
            return None
        if not self._is_supported(stock_code):
            return None
        ticker = stock_code.strip().upper()
        try:
            payload = self._request_json(
                f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
            )
        except DataFetchError as e:
            logger.warning("[Massive] 获取 %s 实时快照失败: %s", ticker, e)
            return None
        snap = (payload or {}).get("ticker") or {}
        if not snap:
            return None
        day = snap.get("day") or {}
        prev_day = snap.get("prevDay") or {}
        last_quote = snap.get("lastTrade") or snap.get("min") or {}

        price = safe_float(last_quote.get("p")) or safe_float(day.get("c")) or safe_float(
            (snap.get("min") or {}).get("c")
        )
        prev_close = safe_float(prev_day.get("c"))
        change_amount = safe_float(snap.get("todaysChange"))
        change_pct = safe_float(snap.get("todaysChangePerc"))
        if change_amount is None and price is not None and prev_close is not None:
            change_amount = price - prev_close
        if change_pct is None and change_amount is not None and prev_close:
            change_pct = (change_amount / prev_close) * 100

        return UnifiedRealtimeQuote(
            code=ticker,
            name=ticker,
            source=RealtimeSource.MASSIVE,
            price=price,
            change_pct=change_pct,
            change_amount=change_amount,
            volume=int(safe_float(day.get("v")) or 0) or None,
            amount=safe_float(day.get("dv")),
            open_price=safe_float(day.get("o")),
            high=safe_float(day.get("h")),
            low=safe_float(day.get("l")),
            pre_close=prev_close,
        )

    # ------------------------------------------------------------------
    # 主指数（仅美股；A 股区域返回 None）
    # ------------------------------------------------------------------
    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        if region != "us" or not _api_key():
            return None
        # Massive/Polygon 美股指数 ticker 形如 I:SPX, I:NDX, I:DJI（部分 tier 不开放指数）
        index_map = [("I:SPX", "S&P 500", "SPX"), ("I:NDX", "NASDAQ 100", "NDX"), ("I:DJI", "Dow Jones", "DJI")]
        results: List[Dict[str, Any]] = []
        for upstream_ticker, name, return_code in index_map:
            try:
                payload = self._request_json(
                    f"/v2/snapshot/locale/us/markets/indices/tickers/{upstream_ticker}"
                )
            except DataFetchError as e:
                logger.debug("[Massive] 指数 %s 跳过: %s", upstream_ticker, e)
                continue
            snap = (payload or {}).get("ticker") or {}
            if not snap:
                continue
            value = safe_float((snap.get("value") if "value" in snap else (snap.get("session") or {}).get("close")))
            change = safe_float(snap.get("change") or (snap.get("session") or {}).get("change"))
            change_pct = safe_float(snap.get("change_percent") or (snap.get("session") or {}).get("change_percent"))
            if value is None:
                continue
            results.append(
                {
                    "code": return_code,
                    "name": name,
                    "current": value,
                    "change": change,
                    "change_pct": change_pct,
                }
            )
        return results or None
