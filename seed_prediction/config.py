from pathlib import Path


from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PATH_DATASET: Path = "seed_prediction/train/dataset/sementes.csv"
    PATH_IAMODEL: Path = "seed_prediction/model/modelo_classificacao.torch"

settings = Settings()
