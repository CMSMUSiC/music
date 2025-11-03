import logging

from rich.console import Console
from rich.logging import RichHandler

# Optional: create a single Console for all output
console = Console()


def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # already configured

    root_logger.setLevel(level)

    rich_handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )

    formatter = logging.Formatter("%(name)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    rich_handler.setFormatter(formatter)

    rich_handler.setFormatter(formatter)

    root_logger.addHandler(rich_handler)
