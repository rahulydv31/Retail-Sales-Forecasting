import logging


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the project.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    return logging.getLogger(name)