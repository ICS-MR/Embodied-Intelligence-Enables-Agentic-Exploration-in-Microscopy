import logging
import sys
from typing import Any, Mapping


RESET = "\033[0m"
DIM = "\033[2m"
LEVEL_COLORS = {
    "DEBUG": "\033[37m",
    "INFO": "\033[36m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m\033[97m",
}
CATEGORY_COLORS = {
    "SYSTEM": "\033[96m",
    "PLANNER": "\033[95m",
    "EXECUTOR": "\033[92m",
    "CHECKER": "\033[93m",
}


class CliCategoryAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Mapping[str, Any]):
        extra = dict(kwargs.get("extra", {}))
        extra.setdefault("cli_category", self.extra["cli_category"])
        kwargs["extra"] = extra
        return msg, kwargs


class CliFormatter(logging.Formatter):
    def __init__(self, use_color: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        category = getattr(record, "cli_category", "SYSTEM").upper()
        level = record.levelname.upper()
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        if self.use_color:
            category_text = f"{CATEGORY_COLORS.get(category, '')}[{category}]{RESET}"
            if level == "INFO":
                level_text = ""
            else:
                level_text = f" {LEVEL_COLORS.get(level, '')}{level}{RESET}"
            return f"{DIM}{timestamp}{RESET} {category_text}{level_text} {message}"

        if level == "INFO":
            return f"{timestamp} [{category}] {message}"
        return f"{timestamp} [{category}] {level} {message}"


def configure_cli_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CliFormatter(use_color=sys.stdout.isatty()))
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)


def get_cli_logger(category: str) -> CliCategoryAdapter:
    return CliCategoryAdapter(logging.getLogger(f"cli.{category.lower()}"), {"cli_category": category.upper()})
