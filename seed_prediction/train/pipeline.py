import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from seed_prediction.model.network_architecture import Modelo
from seed_prediction.config import settings


def train():

    dados = pd.read_csv(settings.PATH_DATASET)

    X = dados.drop(["Espécie"], axis=1).values
    y = dados["Espécie"].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2)

    X_treino = torch.FloatTensor(X_treino)
    X_teste = torch.FloatTensor(X_teste)
    y_treino = torch.LongTensor(y_treino)
    y_teste = torch.LongTensor(y_teste)

    modelo_classificacao = Modelo()

    funcao_objetivo = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo_classificacao.parameters(), lr=0.01)

    epocas = 100
    custos = []

    for _ in range(epocas):
        y_predito = modelo_classificacao.forward(X_treino)
        custo = funcao_objetivo(y_predito, y_treino)
        custos.append(custo)

        otimizador.zero_grad()
        custo.backward()
        otimizador.step()

    print(f"Loss: {custos[-1]:e}")

    preds = []
    with torch.no_grad():
        for val in X_teste:
            y_predito = modelo_classificacao.forward(val)
            preds.append(y_predito.argmax().item())

    df = pd.DataFrame({"Y": y_teste, "YHat": preds})

    df["correto"] = [
        1
        if corr == pred else 0 for corr, pred in zip(df["Y"], df["YHat"])
    ]

    print("{:.2%}".format(sum((df["correto"] == 1)) / df.shape[0]))

    torch.save(modelo_classificacao.state_dict(), settings.PATH_IAMODEL)
