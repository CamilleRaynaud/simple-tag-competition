import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from pathlib import Path
import random

# ===== Hyperparameters =====
OBS_DIM = 14
ACTION_DIM = 5
HIDDEN_DIM = 256
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
ENTROPY_COEF = 0.01
MAX_EPISODES = 50000  # grande quantité pour convergence optimale
MAX_CYCLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 5

# ===== Actor-Critic Networks =====
class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    """Centralized critic sees all predators' observations concatenated"""
    def __init__(self, obs_dim=OBS_DIM*3, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

# ===== PPO Update =====
def ppo_update(actor, critic, optimizer_actor, optimizer_critic, states, actions, returns, old_logits):
    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    old_logits = torch.tensor(old_logits, dtype=torch.float32, device=DEVICE)

    for _ in range(EPOCHS):
        logits = actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        ratio = torch.exp(dist.log_prob(actions) - old_dist.log_prob(actions))
        advantage = returns - critic(states).squeeze()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(critic(states).squeeze(), returns)
        entropy = dist.entropy().mean()
        loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        loss.backward()
        optimizer_actor.step()
        optimizer_critic.step()

# ===== Training Loop =====
def train():
    env = simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=2,
        max_cycles=MAX_CYCLES, continuous_actions=False
    )

    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR)

    for episode in range(MAX_EPISODES):
        obs = env.reset()
        done = {agent: False for agent in env.agents}
        memory = {'states': [], 'actions': [], 'rewards': [], 'logits': []}

        while env.agents:
            actions_dict = {}
            predator_obs = []
            for agent_id in env.agents:
                o = obs[agent_id]
                if "adversary" in agent_id:
                    predator_obs.append(o)

            # Centralized critic state
            central_state = np.concatenate(predator_obs)
            central_state_tensor = torch.tensor(central_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Each predator takes action
            for i, agent_id in enumerate(env.agents):
                o = obs[agent_id]
                if "adversary" in agent_id:
                    obs_tensor = torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    logits = actor(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                    memory['states'].append(o)
                    memory['actions'].append(action)
                    memory['logits'].append(logits.detach().cpu().squeeze().numpy())
                    actions_dict[agent_id] = action
                else:
                    actions_dict[agent_id] = random.randint(0, ACTION_DIM-1)

            obs, rewards, terminations, truncations, infos = env.step(actions_dict)

            # Accumulate predator rewards
            for agent_id, r in rewards.items():
                if "adversary" in agent_id:
                    memory['rewards'].append(r)

            if all(list(terminations.values())) or all(list(truncations.values())):
                break

        # Compute returns
        returns = []
        R = 0
        for r in reversed(memory['rewards']):
            R = r + GAMMA * R
            returns.insert(0, R)

        # PPO update
        ppo_update(actor, critic, optimizer_actor, optimizer_critic,
                   memory['states'], memory['actions'], returns, memory['logits'])

        if episode % 100 == 0:
            print(f"Episode {episode}/{MAX_EPISODES} completed")

    # Save actor for submission
    save_path = Path("predator_model.pth")
    torch.save(actor.state_dict(), save_path)
    print(f"MAPPO actor saved to {save_path}")

if __name__ == "__main__":
    train()
