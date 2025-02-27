import atexit as _atexit
import os
import sys

from loguru import logger as loguru_logger

LOGURU_LOG_LEVEL = os.getenv("POLARIS_ASAP_ADMET_LOG_LEVEL", "INFO")
LOGURU_LOG_TO_FILE = bool(os.getenv("POLARIS_ASAP_ADMET_LOG_TO_FILE", False))
__all__ = ["logger"]

LOGURU_LOG_FORMAT = (
    "<level>{level: <8}| </level>"
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}| </green>"
    "<cyan>{name}:{line:<4d}| </cyan>"
    "<level>{message}</level>"
)

# Maybe gonna need another handler at some point, but this is fine for now
LOGURU_HANDLER = {
    "sink": sys.stdout,
    "level": LOGURU_LOG_LEVEL,
    "colorize": True,
    "format": LOGURU_LOG_FORMAT,
    # The internet claims diagnose and backtrace may deadlock.  Maybe?  Don't know.
    # But, official docs state that diagnose may leak info in prod, so beware
    "backtrace": True,
    "diagnose": False,
}

loguru_logger.configure(handlers=[LOGURU_HANDLER])

logger = loguru_logger

if LOGURU_LOG_TO_FILE:
    logger.add(
        "log/polaris-asap-admet.log",
        level=LOGURU_LOG_LEVEL,
        rotation="1 day",
        compression="zip",
        serialize=False,
        format="{time} | {level} | {name}:{function}:{line} - {message}",
    )

# Clean up the logger on exit
_atexit.register(logger.remove)
