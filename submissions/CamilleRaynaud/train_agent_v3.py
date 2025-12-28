import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from pathlib import Path
import random
from collections import deque

# ===== Hyperparameters =====
HIDDEN_DIM = 256
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
ENTROPY_COEF_INIT = 0.05
ENTROPY_DECAY = 0.995
MAX_EPISODES = 50000
MAX_CYCLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 64
ACTION_DIM = 5
SAVE_EVERY = 500  # checkpoints

# ===== Actor & Critic =====
class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
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
    def __init__(self, obs_dim, hidden_dim=HIDDEN_DIM):
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

# ===== PPO update =====
def ppo_update(actor, critic, optimizer_actor, optimizer_critic,
               states_actor, states_critic, actions, returns, old_logits,
               entropy_coef):

    states_actor = torch.tensor(np.array(states_actor), dtype=torch.float32, device=DEVICE)
    states_critic = torch.tensor(np.array(states_critic), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(np.array(actions), dtype=torch.int64, device=DEVICE)
    returns = torch.tensor(np.array(returns), dtype=torch.float32, device=DEVICE)
    old_logits = torch.tensor(np.array(old_logits), dtype=torch.float32, device=DEVICE)

    n_samples = len(states_actor)
    indices = np.arange(n_samples)

    for _ in range(EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, n_samples, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_idx = indices[start:end]

            batch_states_actor = states_actor[batch_idx]
            batch_states_critic = states_critic[batch_idx]
            batch_actions = actions[batch_idx]
            batch_returns = returns[batch_idx]
            batch_old_logits = old_logits[batch_idx]

            logits = actor(batch_states_actor)
            dist = torch.distributions.Categorical(logits=logits)
            old_dist = torch.distributions.Categorical(logits=batch_old_logits)

            ratio = torch.exp(dist.log_prob(batch_actions) - old_dist.log_prob(batch_actions))
            advantage = batch_returns - critic(batch_states_critic).squeeze()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(critic(batch_states_critic).squeeze(), batch_returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()

# ===== Training loop =====
def train():
    env = simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=2,
        max_cycles=MAX_CYCLES, continuous_actions=False
    )

    obs_dict, infos = env.reset()
    sample_agent = [a for a in obs_dict.keys() if "adversary" in a][0]
    obs_dim = len(obs_dict[sample_agent])

    actor = Actor(obs_dim=obs_dim).to(DEVICE)
    critic = Critic(obs_dim=obs_dim*3).to(DEVICE)
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR)
    scheduler_actor = optim.lr_scheduler.CosineAnnealingLR(optimizer_actor, T_max=MAX_EPISODES)
    scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(optimizer_critic, T_max=MAX_EPISODES)

    entropy_coef = ENTROPY_COEF_INIT
    reward_window = deque(maxlen=100)
    best_avg_reward = -float('inf')

    for episode in range(1, MAX_EPISODES+1):
        obs_dict, infos = env.reset()
        memory = {'states_actor': [], 'states_critic': [], 'actions': [], 'rewards': [], 'logits': []}

        while env.agents:
            actions_dict = {}
            predator_obs_list = []

            for agent_id in env.agents:
                o = obs_dict[agent_id]
                if "adversary" in agent_id:
                    predator_obs_list.append(o)

            central_state = np.concatenate(predator_obs_list)

            for agent_id in env.agents:
                o = obs_dict[agent_id]
                if "adversary" in agent_id:
                    obs_tensor = torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    logits = actor(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()

                    memory['states_actor'].append(o)
                    memory['states_critic'].append(central_state)
                    memory['actions'].append(action)
                    memory['logits'].append(logits.detach().cpu().squeeze().numpy())
                    actions_dict[agent_id] = action
                else:
                    actions_dict[agent_id] = random.randint(0, ACTION_DIM-1)

            obs_dict, rewards, terminations, truncations, infos = env.step(actions_dict)

            for agent_id, r in rewards.items():
                if "adversary" in agent_id:
                    memory['rewards'].append(r)

            if all(list(terminations.values())) or all(list(truncations.values())):
                break

        returns = []
        R = 0
        for r in reversed(memory['rewards']):
            R = r + GAMMA * R
            returns.insert(0, R)
        reward_window.extend(memory['rewards'])
        avg_reward = np.mean(reward_window)

        ppo_update(actor, critic, optimizer_actor, optimizer_critic,
                   memory['states_actor'], memory['states_critic'],
                   memory['actions'], returns, memory['logits'],
                   entropy_coef=entropy_coef)

        scheduler_actor.step()
        scheduler_critic.step()
        entropy_coef *= ENTROPY_DECAY

        if episode % 50 == 0:
            print(f"Episode {episode}/{MAX_EPISODES} | Avg Reward (last 100): {avg_reward:.2f}")

        if episode % SAVE_EVERY == 0 or avg_reward > best_avg_reward:
            best_avg_reward = max(best_avg_reward, avg_reward)
            save_path = Path(f"predator_model_best.pth")
            torch.save(actor.state_dict(), save_path)
            print(f"Model saved at episode {episode} | Best avg reward: {best_avg_reward:.2f}")

        # Early stopping if plateau
        if avg_reward >= 1200:  # objectif
            print(f"Target reached! Episode {episode} | Avg reward: {avg_reward:.2f}")
            break

    print("Training completed. Best avg reward:", best_avg_reward)

if __name__ == "__main__":
    train()
