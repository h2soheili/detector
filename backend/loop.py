import asyncio
from asyncio import AbstractEventLoop

from backend.global_log import structlog

logger = structlog.get_logger(__name__)


def get_loop() -> AbstractEventLoop:
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        logger.error("get_loop", error=e)
        loop = asyncio.new_event_loop()
    finally:
        return loop
