import logging
import os

from logging import FileHandler, StreamHandler, Formatter
from os import PathLike

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO
LOG_ENCODING = "utf-8"


# handling external_modules logging to file
external_loggers = list()
external_modules = []

for module in external_modules:
    os.makedirs("logs", exist_ok=True)
    ext_logger = logging.getLogger(module)
    ext_logger.setLevel(LOG_LEVEL)
    ext_logger_formatter = Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    ext_logger_stream_handler = StreamHandler()
    ext_logger_stream_handler.setLevel(LOG_LEVEL)
    ext_logger_stream_handler.setFormatter(ext_logger_formatter)

    ext_logger_file_handler = FileHandler(
        os.path.join("logs", f"{module}.log"), encoding=LOG_ENCODING
    )
    ext_logger_file_handler.setLevel(LOG_LEVEL)
    ext_logger_file_handler.setFormatter(ext_logger_formatter)

    ext_logger.addHandler(ext_logger_stream_handler)
    ext_logger.addHandler(ext_logger_file_handler)

    external_loggers.append(ext_logger)


# handling remaining logging by function call in respective module
def init_logging(
    name: str,
    out_file: PathLike | None = None,
    encoding: str = "utf-8",
    format: str = "%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    level: int = logging.INFO,
):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(level)
        formatter = Formatter(fmt=format, datefmt=datefmt)
        stream_handler = StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if out_file:
            os.makedirs("logs", exist_ok=True)
            file_handler = FileHandler(
                filename=os.path.join("logs", out_file), encoding=encoding
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
