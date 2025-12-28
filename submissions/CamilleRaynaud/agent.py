import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim=5, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class StudentAgent:
    def __init__(self):
            model_path = Path(__file__).parent / "predator_actor_v2.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file: {model_path}")

            state = torch.load(model_path, map_location="cpu")

            # lire la vraie dimension d'entrée du modèle
            self.obs_dim = state["net.0.weight"].shape[1]

            self.actor = ActorNet(self.obs_dim, 5, 256)
            self.actor.load_state_dict(state)
            self.actor.eval()

    def get_action(self, observation, agent_id):
            x = np.asarray(observation, dtype=np.float32)

            # tronquage / pad minimal mais cohérent
            if x.shape[0] > self.obs_dim:
                x = x[:self.obs_dim]
            elif x.shape[0] < self.obs_dim:
                pad = np.zeros(self.obs_dim - x.shape[0], dtype=np.float32)
                x = np.concatenate([x, pad])

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = self.actor(x)
                action = torch.argmax(logits, dim=1).item()

            return int(action)
