
from fastapi import FastAPI

from sementes_ai.routes import router

app = FastAPI()

app.include_router(router=router)
