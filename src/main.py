import uvicorn

if __name__ == '__main__':
    # Run FastAPI in the main thread
    uvicorn.run(
        host="0.0.0.0",
        port=8000,
        log_level="info",
        app="app:app",
        timeout_keep_alive=9999,
        ws_ping_timeout=9999,
        limit_concurrency=9999,
        reload=True
    )