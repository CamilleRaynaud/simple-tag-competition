import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from pettingzoo.mpe import simple_tag_v3


class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim=5, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def select_action(policy, obs):
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits = policy(x)
    probs = torch.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def main():
    save_path = Path("submissions/CamilleRaynaud/predator_actor_final.pth")

    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False,
    )

    observations, _ = env.reset()
    predator_ids = [i for i in observations if "adversary" in i]

    sample_obs = observations[predator_ids[0]]
    obs_dim = sample_obs.shape[0]
    action_dim = 5

    policy = ActorNet(obs_dim, action_dim, 256)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    gamma = 0.99
    num_episodes = 4000

    print(f"Training with obs_dim={obs_dim}")

    for episode in range(num_episodes):
        log_probs = {pid: [] for pid in predator_ids}
        rewards = {pid: [] for pid in predator_ids}

        observations, _ = env.reset()

        done = False
        while not done:
            actions = {}

            for pid in predator_ids:
                a, logp = select_action(policy, observations[pid])
                actions[pid] = a
                log_probs[pid].append(logp)

            # random actions for prey / obstacles
            for aid in observations:
                if aid not in actions:
                    actions[aid] = np.random.randint(5)

            observations, rewards_step, terminations, truncations, _ = env.step(actions)

            for pid in predator_ids:
                rewards[pid].append(rewards_step[pid])

            done = any(terminations.values()) or any(truncations.values())

        # REINFORCE update
        loss = 0
        for pid in predator_ids:
            G = 0
            returns = []
            for r in reversed(rewards[pid]):
                G = r + gamma * G
                returns.insert(0, G)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            for logp, Gt in zip(log_probs[pid], returns):
                loss += -logp * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 200 == 0:
            print(f"Episode {episode+1}/{num_episodes} — loss={loss.item():.3f}")

    env.close()

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
        },
        save_path,
    )

    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
