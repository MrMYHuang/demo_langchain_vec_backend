import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "demo_langchain_vec:app",
        host="0.0.0.0", port=8000, reload=True,
        reload_includes=["server.py", "demo_rag.py"], reload_excludes="*.py")