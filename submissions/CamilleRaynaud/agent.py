import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_DIM = 5

class Actor(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, ACTION_DIM)
        )

    def forward(self, x):
        return self.net(x)

class StudentAgent:
    def __init__(self):
        root = Path(__file__).parent

        metadata = json.loads((root / "predator_metadata.json").read_text())
        self.obs_dim = metadata["obs_dim"]

        self.actor = Actor(self.obs_dim).to(DEVICE)
        self.actor.load_state_dict(torch.load(root / "predator_actor.pth", map_location=DEVICE))
        self.actor.eval()

    def get_action(self, observation, agent_id):
        obs = np.asarray(observation, dtype=np.float32)

        # HARD CHECK — no silent mismatch
        assert len(obs) == self.obs_dim, \
            f"Unexpected obs_dim {len(obs)} != {self.obs_dim}"

        x = torch.tensor(obs, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(x)
            action = torch.argmax(logits, dim=1).item()

        return int(action)
