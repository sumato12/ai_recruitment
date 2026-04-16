from recruitment_api.app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000)

