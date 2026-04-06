import json
import os
import time


_log_f = None
_metrics_f = None
_log_was_empty = False


def ts() -> str:
    return time.strftime("%d/%m/%Y | %H:%M:%S")


def setup_logging(log_path: str, metrics_path: str, mode: str) -> None:
    global _log_f, _metrics_f, _log_was_empty

    if mode == "new":
        _log_was_empty = True
    else:
        try:
            _log_was_empty = os.path.getsize(log_path) == 0
        except FileNotFoundError:
            _log_was_empty = True

    _log_f = open(log_path, "w" if mode == "new" else "a", encoding="utf-8", buffering=1)
    _metrics_f = open(
        metrics_path,
        "w" if mode == "new" else "a",
        encoding="utf-8",
        buffering=1,
    )


def console(msg: str) -> None:
    print(f"{ts()} | {msg}", flush=True)


def log(msg: str) -> None:
    line = f"{ts()} | {msg}"
    print(line, flush=True)
    _log_f.write(line + "\n")
    _log_f.flush()


def log_startup(msg: str) -> None:
    if _log_was_empty:
        log(msg)
    else:
        console(msg)


def metrics_write(obj: dict, mode: str) -> None:
    obj.setdefault("ts", ts())
    obj.setdefault("mode", mode)
    _metrics_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _metrics_f.flush()


def end_logging() -> None:
    global _log_f, _metrics_f
    if _log_f is not None:
        _log_f.flush()
        _log_f.close()
        _log_f = None
    if _metrics_f is not None:
        _metrics_f.flush()
        _metrics_f.close()
        _metrics_f = None