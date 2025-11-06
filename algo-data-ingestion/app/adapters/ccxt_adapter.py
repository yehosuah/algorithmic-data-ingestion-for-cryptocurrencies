# Status: ✅ Exists & wired into async pipeline

import asyncio
import contextlib
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

import ccxt.async_support as ccxt
import pandas as pd

from app.common.time_norm import add_dt_partition, coerce_schema, to_utc_dt


def _since_to_millis(since: Optional[Union[int, float, datetime]]) -> Optional[int]:
    """Convert various `since` representations to milliseconds since epoch.

    Accepts None, seconds (int/float), milliseconds (int >= 1e12), or datetime.
    Returns None if input is None.
    """
    if since is None:
        return None
    if isinstance(since, (int, float)):
        val = float(since)
        # heuristic: treat really large numbers as ms already
        return int(val if val >= 1e12 else val * 1000)
    if isinstance(since, datetime):
        return int(since.timestamp() * 1000)
    # Last resort: try pandas parsing
    try:
        return int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    except Exception:
        raise TypeError(f"Unsupported type for since: {type(since)!r}")


async def _retry_async(func, *args, retries: int = 3, backoff_factor: float = 1.0, **kwargs):
    """Retry an async callable with exponential backoff.

    Args:
        func: async function/coroutine to call
        retries: total attempts (including the first)
        backoff_factor: base backoff seconds (exponential)
    """
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            try:
                logger.warning(
                    f"CCXT call {getattr(func, '__name__', str(func))} failed on attempt {attempt}/{retries}: {e}"
                )
            except Exception:
                pass
            if attempt < retries:
                await asyncio.sleep(backoff_factor * (2 ** (attempt - 1)))
    # last attempt; let exception propagate if it fails
    return await func(*args, **kwargs)

logger = logging.getLogger("app.adapters.ccxt_adapter")


def _env_to_bool(value: Optional[Union[str, int, float, bool]]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"0", "false", "no", "off"}:
        return False
    return True

class CCXTAdapter:
    def __init__(
        self,
        exchange_id: str = "binance",
        *,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        password: Optional[str] = None,
        sandbox: Optional[bool] = None,
    ):
        self.exchange_id = exchange_id
        exchange_cls = getattr(ccxt, exchange_id)
        prefix = exchange_id.upper().replace("-", "_")

        api_key = api_key or os.getenv(f"{prefix}_API_KEY") or os.getenv("EXCHANGE_API_KEY")
        secret = secret or os.getenv(f"{prefix}_API_SECRET") or os.getenv("EXCHANGE_API_SECRET")
        password = password or os.getenv(f"{prefix}_API_PASSWORD") or os.getenv("EXCHANGE_API_PASSWORD")

        sandbox_env = os.getenv(f"{prefix}_SANDBOX_MODE")
        if sandbox_env is None:
            sandbox_env = os.getenv("EXCHANGE_SANDBOX_MODE")
        sandbox = sandbox if sandbox is not None else _env_to_bool(sandbox_env)

        config: Dict[str, Any] = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if secret:
            config["secret"] = secret
        if password:
            config["password"] = password

        self.client = exchange_cls(config)
        self._markets_loaded = False

        if sandbox:
            with contextlib.suppress(Exception):
                self.client.set_sandbox_mode(True)
                logger.debug("Sandbox mode enabled for %s", exchange_id)

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches the ticker for a given symbol via CCXT
        Returns the raw JSON dict from the exchange
        """
        return await _retry_async(self.client.fetch_ticker, symbol)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[Any] = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.
        Returns a DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        since_ms = _since_to_millis(since)
        ohlcv = await _retry_async(self.client.fetch_ohlcv, symbol, timeframe, since_ms, limit)
        df = pd.DataFrame(ohlcv, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        # Make timestamp tz-aware UTC, add identifying columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['symbol'] = symbol
        df['exchange'] = self.exchange_id
        df['timeframe'] = timeframe

        # Enforce schema and add dt partition
        schema = {
            'timestamp': 'datetime64[ns, UTC]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            'symbol': 'string',
            'exchange': 'string',
            'timeframe': 'string',
        }
        df = coerce_schema(df, schema)
        add_dt_partition(df, ts_col='timestamp')
        return df

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch current order book snapshot for a symbol.
        Returns a DataFrame with bids and asks.
        """
        ob = await _retry_async(self.client.fetch_order_book, symbol, limit)
        bids = pd.DataFrame(ob['bids'], columns=['price', 'amount'])
        bids['side'] = 'bid'
        asks = pd.DataFrame(ob['asks'], columns=['price', 'amount'])
        asks['side'] = 'ask'
        # Stamp snapshot time and identifiers, normalize schema
        now_utc = pd.Timestamp.now(tz='UTC')
        bids['timestamp'] = now_utc
        asks['timestamp'] = now_utc
        bids['symbol'] = symbol
        asks['symbol'] = symbol
        bids['exchange'] = self.exchange_id
        asks['exchange'] = self.exchange_id

        df = pd.concat([bids, asks], ignore_index=True)
        schema = {
            'timestamp': 'datetime64[ns, UTC]',
            'price': 'float64',
            'amount': 'float64',
            'side': 'string',
            'symbol': 'string',
            'exchange': 'string',
        }
        df = coerce_schema(df, schema)
        add_dt_partition(df, ts_col='timestamp')
        return df

    async def watch_ticker(
        self,
        symbol: str,
        callback: callable
    ):
        """
        Stream live ticker updates via WebSocket.
        Calls callback with each ticker update dict.
        """
        await self.client.load_markets()
        while True:
            ticker = await self.client.watch_ticker(symbol)
            callback(ticker)

    async def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balances.
        Returns the raw balance dict.
        """
        return await _retry_async(self.client.fetch_balance)

    async def list_symbols(self) -> list:
        """
        Return list of symbols available on the exchange.
        """
        markets = await _retry_async(self.client.load_markets)
        return list(markets.keys())

    async def close(self):
        await self.client.close()

    async def ensure_markets(self) -> None:
        """
        Lazily load market metadata (precision, limits) required for order placement.
        """
        if self._markets_loaded:
            return
        await _retry_async(self.client.load_markets)
        self._markets_loaded = True

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a market order. The caller is responsible for ensuring amount precision.
        """
        await self.ensure_markets()
        return await _retry_async(
            self.client.create_order,
            symbol,
            "market",
            side,
            float(amount),
            None,
            params or {},
        )

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """
        Round the provided base amount using the exchange precision metadata.
        """
        if not self._markets_loaded:
            raise RuntimeError("Market metadata not loaded; call ensure_markets() first")
        return float(self.client.amount_to_precision(symbol, amount))

# Convenience function if you prefer not to manage client lifecycles
async def get_ticker_raw(symbol: str) -> Dict[str, Any]:
    adapter = CCXTAdapter()
    try:
        return await adapter.fetch_ticker(symbol)
    finally:
        await adapter.close()
