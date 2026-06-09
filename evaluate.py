import torch

from environment import PaddleEnvironment
from ddqn_agent import QNetwork


def LoadAgent():

    Model = QNetwork(5, 3)

    Model.load_state_dict(
        torch.load(
            "models/ddqn_model.pth",
            map_location="cpu"
        )
    )

    Model.eval()

    return Model