import logging
import os
import sys
from pathlib import Path

_file_handler_ref = None


class ExcludeOptunaFilter(logging.Filter):
    """
    Filtra ed esclude i log dei singoli trial di Optuna dal file di log principale.
    """

    def filter(self, record):
        return "[Optuna Trial]" not in record.getMessage()


_file_handler_ref = None
_main_handler_ref = None 


def setup_logger(
        level=logging.INFO, 
        run_log_file_path: str | Path | None = None,
        handler_mode: str = "w",
        main_log_file_path: str | Path | None = None
    ) -> None:
    global _file_handler_ref, _main_handler_ref

    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
    logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
    logging.getLogger("alembic.runtime.migration").disabled = True
    logging.getLogger("alembic.runtime").disabled = True
    logging.getLogger("alembic").disabled = True

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if run_log_file_path is not None:
        path = Path(run_log_file_path)
        os.makedirs(path.parent, exist_ok=True)

        file_handler = logging.FileHandler(path, mode=handler_mode)
        file_handler.addFilter(ExcludeOptunaFilter())

        handlers.append(file_handler)
        _file_handler_ref = file_handler

    if main_log_file_path is not None:
        path = Path(main_log_file_path)
        os.makedirs(path.parent, exist_ok=True)

        file_main_handler = logging.FileHandler(path, mode="a")
        file_main_handler.addFilter(ExcludeOptunaFilter())

        handlers.append(file_main_handler)
        _main_handler_ref = file_main_handler
    

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def restore_logging_after_mlflow(level=logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    global _file_handler_ref, _main_handler_ref
    for handler in [_file_handler_ref, _main_handler_ref]:
        if handler is not None:
            if handler.stream is None or handler.stream.closed:
                handler.stream = handler._open()
            if handler not in root.handlers:
                root.addHandler(handler)
    for h in root.handlers:
        h.setFormatter(formatter)
