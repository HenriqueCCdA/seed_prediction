
from rich.console import Console
from typer import Typer

from seed_prediction.train.pipeline import train as train_
from seed_prediction.config import settings

console = Console()

app_cli = Typer(add_completion=False)

@app_cli.command()
def train():
    """Treinamento da MLP"""
    train_()

@app_cli.command()
def pred():
    """"""

@app_cli.command()
def show_config():
    """Mostra as configuras"""
    console.print(settings)
