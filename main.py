import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # Changed to import string format
        host="0.0.0.0",
        port=8000,
        reload=True
    )