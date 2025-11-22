from fastapi import FastAPI
from backend.blip2 import generate_caption

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Upender! Your backend is working!"}