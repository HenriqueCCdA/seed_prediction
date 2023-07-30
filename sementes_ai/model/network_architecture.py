import torch
import torch.nn as nn
import torch.nn.functional as F

from sementes_ai.config import settings

class Modelo(nn.Module):

    def __init__(
        self,
        entrada: int=7,
        camada_escondida1: int=14,
        camada_escondida2: int=49,
        saida:int=3
    ):
        super().__init__()
        self.fc1 = nn.Linear(entrada, camada_escondida1)
        self.fc2 = nn.Linear(camada_escondida1, camada_escondida2)
        self.out = nn.Linear(camada_escondida2, saida)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


model = Modelo()
state_dict = torch.load(settings.PATH_IAMODEL)
model.load_state_dict(state_dict)
model.eval()
