# -*- coding: utf-8 -*-
"""
===================================
EodhdFetcher - EODHD 数据源 (港股 + 沪深 A 股日线, Priority 0)
===================================

EODHD (https://eodhd.com/) 提供港股与沪深 A 股的 EOD 日线及港股延时报价。
在 efinance / akshare 的港股接口不可用或频繁超时的环境下, 可作为稳定的日线来源。

定位与边界:
- 日线服务港股 (.HK) 与沪深 A 股 (沪 .SHG / 深 .SHE); 美股、北交所等代码
  立即抛 DataFetchError, 让位下一数据源;
- 另实现港股 get_realtime_quote / get_stock_name 方法 (当前管理器的港股
  实时链路未调用 get_realtime_quote);
- 当日缺根补齐: end_date 含今日且 EOD 尚未出当日收盘根时, 用 /real-time
  补一根 (失败静默, 不阻塞);
- 注册在数据源列表头 (priority 0); 未配置 key 时每次调用立即失败并让位;
- API key: 环境变量 EODHD_API_KEY; 未设置时可用 EODHD_ENV_FILE 指向另一份
  含 EODHD_API_KEY 的 .env (相对路径以本项目根目录为基准);
- 价格为 EODHD 原始 OHLC (未复权); 历史数据中的 999999 哨兵行
  (close/open/high/low ≥ 999000) 会被丢弃。
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, _is_hk_market
from .realtime_types import UnifiedRealtimeQuote, RealtimeSource

logger = logging.getLogger(__name__)

_EODHD_BASE = "https://eodhd.com/api"
_SENTINEL = 999000.0          # EODHD 历史数据哨兵值下界 (实际哨兵 999999)
_HTTP_TIMEOUT = 15


def _load_api_key() -> Optional[str]:
    """EODHD_API_KEY 环境变量优先; 缺失时读 EODHD_ENV_FILE 指向的 .env
    (相对路径以本项目根目录为基准)。"""
    key = (os.environ.get("EODHD_API_KEY") or "").strip()
    if key:
        return key
    env_file = (os.environ.get("EODHD_ENV_FILE") or "").strip()
    if not env_file:
        return None
    try:
        env_path = Path(env_file).expanduser()
        if not env_path.is_absolute():
            env_path = Path(__file__).resolve().parents[1] / env_path
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if line.startswith("EODHD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    except Exception:
        pass
    return None


def _hk_symbol(stock_code: str) -> Optional[str]:
    """'hk00700' / '00700' / '700.hk' → EODHD '0700.HK' (4位补零)。非港股 → None。"""
    code = str(stock_code or "").strip().lower()
    if code.endswith(".hk"):
        code = code[:-3]
    if code.startswith("hk"):
        code = code[2:]
    if not code.isdigit() or not (1 <= len(code) <= 5):
        return None
    return f"{int(code):04d}.HK"


def _cn_symbol(stock_code: str) -> Optional[str]:
    """沪深 6 位代码 → EODHD 符号: 沪(6/9开头)→.SHG, 深(0/2/3开头)→.SHE。
    北交所(4/8开头)/其他 EODHD 无数据 → None 让位。"""
    code = str(stock_code or "").strip().upper()
    for pre in ("SH", "SZ"):
        if code.startswith(pre):
            code = code[len(pre):]
    code = code.lstrip(".")
    if not code.isdigit() or len(code) != 6:
        return None
    if code[0] in ("6", "9"):
        return f"{code}.SHG"
    if code[0] in ("0", "2", "3"):
        return f"{code}.SHE"
    return None


def _eod_symbol(stock_code: str) -> Optional[str]:
    """日线可服务的市场: 港股 + 沪深A股。其余 None (让位下一数据源)。"""
    if _is_hk_market(stock_code):
        return _hk_symbol(stock_code)
    return _cn_symbol(stock_code)


class EodhdFetcher(BaseFetcher):
    """EODHD 数据源: 港股 / 沪深 A 股日线; 港股实时报价与名称。"""

    name = "EodhdFetcher"
    priority = 0

    def __init__(self):
        self._api_key = _load_api_key()
        if not self._api_key:
            logger.warning("[EodhdFetcher] 未配置 EODHD_API_KEY, EODHD 数据源将让位其他数据源")

    # ── HTTP ────────────────────────────────────────────────
    def _get_json(self, path: str, params: Dict[str, Any]) -> Any:
        if not self._api_key:
            raise DataFetchError("[EodhdFetcher] EODHD_API_KEY 缺失")
        qs = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        url = f"{_EODHD_BASE}/{path}?api_token={self._api_key}&fmt=json&{qs}"
        req = Request(url, headers={"User-Agent": "dsa-eodhd-fetcher/1.0"})
        started = time.time()
        with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        logger.debug(f"[EodhdFetcher] GET {path} 耗时 {time.time()-started:.2f}s")
        return data

    # ── 日线 (BaseFetcher 模板挂钩) ─────────────────────────
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = _eod_symbol(stock_code)
        if symbol is None:
            raise DataFetchError(
                f"[EodhdFetcher] 仅支持港股/沪深A股, 让位下一数据源: {stock_code}")
        rows = self._get_json(f"eod/{symbol}",
                              {"from": start_date, "to": end_date, "period": "d"})
        if not isinstance(rows, list) or not rows:
            raise DataFetchError(f"[EodhdFetcher] {symbol} 无数据 ({start_date}~{end_date})")
        # 当日缺根补齐: end_date 含今日而 EOD 未出今日行 → /real-time 补一根
        try:
            today = time.strftime("%Y-%m-%d")
            if end_date >= today and rows and str(rows[-1].get("date")) < today:
                rt = self._get_json(f"real-time/{symbol}", {})
                ts = rt.get("timestamp") if isinstance(rt, dict) else None
                rt_date = time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else None
                close = rt.get("close") if isinstance(rt, dict) else None
                if rt_date == today and isinstance(close, (int, float)) and close < _SENTINEL:
                    rows.append({"date": today, "open": rt.get("open"),
                                 "high": rt.get("high"), "low": rt.get("low"),
                                 "close": close, "volume": rt.get("volume") or 0})
        except Exception as e:
            logger.debug(f"[EodhdFetcher] {symbol} 当日补根失败(忽略): {e}")
        df = pd.DataFrame(rows)
        # 哨兵清洗 (EODHD 历史缺口期以 999999 填充)
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df = df[pd.to_numeric(df[col], errors="coerce") < _SENTINEL]
        if df.empty:
            raise DataFetchError(f"[EodhdFetcher] {symbol} 哨兵清洗后无有效行")
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()
        # EODHD 列: date/open/high/low/close/adjusted_close/volume (已是小写)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["pct_chg"] = (df["close"].pct_change() * 100).fillna(0).round(2)
        # EODHD 不提供成交额 → 估算 (与 YfinanceFetcher 同约定)
        df["amount"] = df["volume"] * df["close"]
        df["code"] = stock_code
        keep = ["code"] + STANDARD_COLUMNS
        return df[[c for c in keep if c in df.columns]]

    # ── 实时报价 ─────────────────────────────────────────────
    def get_realtime_quote(self, stock_code: str, source: str = "eodhd"
                           ) -> Optional[UnifiedRealtimeQuote]:
        symbol = _hk_symbol(stock_code) if _is_hk_market(stock_code) else None
        if symbol is None:
            return None
        try:
            d = self._get_json(f"real-time/{symbol}", {})
        except Exception as e:
            logger.warning(f"[EodhdFetcher] 实时报价失败 {symbol}: {e}")
            return None
        if not isinstance(d, dict) or d.get("close") in (None, "NA"):
            return None

        def _f(k):
            v = d.get(k)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return UnifiedRealtimeQuote(
            code=stock_code,
            name="",
            source=getattr(RealtimeSource, "EODHD", RealtimeSource.FALLBACK),
            price=_f("close"),
            change_pct=_f("change_p"),
            change_amount=_f("change"),
            volume=int(_f("volume") or 0) or None,
            open_price=_f("open"),
            high=_f("high"),
            low=_f("low"),
            pre_close=_f("previousClose"),
        )

    # ── 名称 (英文名兜底, 中文名仍靠 STOCK_NAME_MAP/其他源) ──
    def get_stock_name(self, stock_code: str) -> Optional[str]:
        symbol = _hk_symbol(stock_code) if _is_hk_market(stock_code) else None
        if symbol is None:
            return None
        try:
            rows = self._get_json(f"search/{symbol}", {"limit": 1})
            if isinstance(rows, list) and rows:
                return str(rows[0].get("Name") or "") or None
        except Exception as e:
            logger.debug(f"[EodhdFetcher] 名称查询失败 {symbol}: {e}")
        return None
