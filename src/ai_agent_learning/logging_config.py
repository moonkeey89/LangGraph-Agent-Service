import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format=LOG_FORMAT,
    )
