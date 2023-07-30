
from fastapi import FastAPI

from seed_prediction.routes import router

app = FastAPI()

app.include_router(router=router)
