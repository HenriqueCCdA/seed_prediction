from turtle import st
from typing import Annotated

from fastapi import Query, HTTPException, status, APIRouter
import torch

from seed_prediction.model.network_architecture import model


router = APIRouter()

@router.get("/")
def pred(X: Annotated[list[float], Query()] = None):

    if not X:
        return {"detail": "Você pode fazer uma predição passando os paramentros por query string"}

    if len(X) != 7:
        raise HTTPException(detail="São necessarios 7 args.", status_code=status.HTTP_400_BAD_REQUEST)

    with torch.no_grad():
        X_tns = torch.FloatTensor(X)
        y_pred = model.forward(X_tns)

    return {"label": y_pred.argmax().item()}
