import numpy as np
import torch as th
from torch.nn import functional as F
from torch import nn

class RND(nn.Module):
    """
    simple RND module for thesis
    """

    def __init__(self, input_dim, output_dim:int = 1, hidden_dim=256, device="cpu"):
        super(RND, self).__init__()

        if not isinstance(input_dim, int):
            input_dim = np.prod(input_dim)

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = th.optim.Adam(self.predictor.parameters(), lr=1e-3)

        self.to(device)

    #adapt this to make it not combine a whole batch into one input
    def forward(self, x):
        pred = self.predictor(x)
        with th.no_grad():
            target = self.target(x)
        return pred, target