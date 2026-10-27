import socket
import sys
import threading
import time
import webbrowser
import uvicorn
from app.core.config import settings
from app.utils.logging import logger


def get_free_port(host: str, start_port: int, attempts: int = 100) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find a free port starting at {start_port}.")


def open_browser(url: str):
    # Timestamp bypasses browser caching of index.html
    webbrowser.open(f"{url}?t={int(time.time())}")


if __name__ == "__main__":
    try:
        port = get_free_port(settings.HOST, settings.PORT)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    browse_host = "localhost" if settings.HOST in ("0.0.0.0", "127.0.0.1", "::") else settings.HOST
    url = f"http://{browse_host}:{port}"
    logger.info("==========================================================")
    logger.info(f"Launching jstRAG on {url}")
    logger.info("==========================================================")

    if settings.OPEN_BROWSER:
        threading.Timer(2.0, open_browser, args=[url]).start()

    # reload=True does not work with PyInstaller executables
    from app.main import app
    uvicorn.run(app, host=settings.HOST, port=port, reload=False)
