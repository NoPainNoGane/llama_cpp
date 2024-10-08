import os
# from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from controller.controller import llama_router



# load_dotenv("../.env")

app = FastAPI(docs_url="/api/docs/")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

PREFIX = "/api/v1"
# Добавление эндпоинтов в приложение
app.include_router(router=llama_router, prefix=PREFIX, tags=["Llama"])


if not os.path.exists('../weights'):
    os.makedirs('../weights')

