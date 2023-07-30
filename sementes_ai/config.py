from pathlib import Path


from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PATH_DATASET: Path = "sementes_ai/train/dataset/sementes.csv"
    PATH_IAMODEL: Path = "sementes_ai/model/modelo_classificacao.torch"

settings = Settings()
