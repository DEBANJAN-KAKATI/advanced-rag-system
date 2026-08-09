import os
import sys
import threading
import webbrowser
import uvicorn
import socket
import time
from app.utils.logging import logger

def get_free_port(start_port=8000, max_port=8100):
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port.")

def open_browser(port):
    # Wait a bit before opening to ensure server is up
    # Add a timestamp to bypass browser caching for index.html
    url = f"http://localhost:{port}?t={int(time.time())}"
    webbrowser.open(url)

if __name__ == "__main__":
    try:
        port = get_free_port()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("==========================================================")
    logger.info(f"Launching jstRAG System live on http://localhost:{port}")
    logger.info("==========================================================")
    
    # We use a timer to open the browser after 2 seconds
    threading.Timer(2.0, open_browser, args=[port]).start()
    
    # Do not use reload=True as it does not work well with PyInstaller executables
    from app.main import app
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
