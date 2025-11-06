from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from prometheus_client import start_http_server

from app.trading.config import TradingConfig
from app.trading.service import TradingService


async def _amain() -> None:
    config = TradingConfig()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [trading] %(message)s",
    )
    if config.metrics_port > 0:
        logging.getLogger("app.trading").info(
            "Starting Prometheus exporter on port %d", config.metrics_port
        )
        start_http_server(config.metrics_port)
    service = TradingService(config)
    await service.start()

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        if not stop_event.is_set():
            logging.getLogger("app.trading").info("Shutdown signal received")
            stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    await stop_event.wait()
    await service.stop()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
