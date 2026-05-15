from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key.startswith("_") and key != "_extra":
                continue
        extra = getattr(record, "_extra", None)
        if isinstance(extra, Mapping):
            payload.update(extra)

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: str = "INFO", log_format: str = "plain") -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestContextFilter())
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] [request_id=%(request_id)s] %(message)s"
            )
        )

    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())


def bind_request_id(request_id: str) -> contextvars.Token[str]:
    return request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    request_id_var.reset(token)


def log_extra(**kwargs: Any) -> dict[str, Any]:
    return {"_extra": kwargs}
