"""Run the actor-crictic algorithm to solve the optimization problem."""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from satellite_rl.reinforcement_learning.environment import SatelliteEnv

logging.basicConfig(level=logging.INFO)
# Set the number of threads for Pytorch
torch.set_num_threads(config.TORCH_NB_THREADS)

# NN layers
HIDDEN_SIZE: int = 128
# Discount factor
GAMMA: float = 0.99
# Learning rates
LR_ACTOR: float = 0.00001
LR_CRITIC: float = 0.0001


class ActorNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int) -> None:
        """Map a state to an action."""
        super().__init__()
        self.fc1_layer: nn.Linear = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mu_out_layer: nn.Sequential = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size), nn.Sigmoid()
        )
        self.sigma_out_layer: nn.Sequential = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size), nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor, torch.distributions.MultivariateNormal]:
        x = self.fc1_layer(x)
        x = self.fc2_layer(x)
        mu: torch.Tensor = self.mu_out_layer(x)
        sigma_diag: torch.Tensor = self.sigma_out_layer(x)
        norm_dist: torch.distributions.MultivariateNormal = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag(sigma_diag)
        )
        x = norm_dist.sample()
        return x.detach(), norm_dist


class CriticNetwork(nn.Module):
    def __init__(self, obs_size: int) -> None:
        """Map a state to a quality-value."""
        super().__init__()
        self.fc1_layer: nn.Linear = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.critic_out_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1_layer(x)
        x = self.sigmoid(x)
        x = self.fc2_layer(x)
        x = self.sigmoid(x)
        x = self.critic_out_layer(x)
        return x


def sample_action(actor: ActorNetwork, env: SatelliteEnv) -> tuple[
    torch.Tensor, torch.Tensor, torch.distributions.MultivariateNormal
]:
    # Scale state
    state: float = 1 / (env.nb_links - 1) * torch.Tensor(env.state.flatten())
    action, norm_dist = actor(state)
    # Clip the action so that the values are in the action space
    action_clipped: torch.Tensor = torch.clip(
        (env.nb_links - 1) * action,
        min=torch.zeros(3, dtype=torch.int),
        max=torch.Tensor([env.nb_links - 1, env.nb_links - 1, env.nb_links - 1]),
    )
    return action, action_clipped, norm_dist


def run_actor_critic(links: list[dict], nb_episodes: int, duration_episode: int,
                     verbose: int = 0) -> tuple[np.ndarray, int, int]:
    """Run the actor-critic algorithm to solve the optimization problem.

    Args:
        links (list): list of satellite links to assign to modems and groups
        nb_episodes (int): number of episodes to run
        duration_episode (int): number of iterations of one episode

    Returns:
        state_min (np.ndarray): the minimal state found
        nb_mod_min (int): the number of modems in the minimal state
        nb_grps_min (int): the number of groups in the minimal group
    """

    env: SatelliteEnv = SatelliteEnv(links)
    actor: ActorNetwork = ActorNetwork(obs_size=2 * len(links), action_size=3)
    critic: CriticNetwork = CriticNetwork(obs_size=2 * len(links))
    # Initiliaze loss
    critic_loss_fn: nn.MSELoss = nn.MSELoss()
    # Initialize optimizers
    actor_optimizer: optim.Adam = optim.Adam(params=actor.parameters(), lr=LR_ACTOR)
    critic_optimizer: optim.Adam = optim.Adam(params=critic.parameters(), lr=LR_CRITIC)
    rewards_list: list = []
    for _ in range(nb_episodes):
        env.reset()
        cumulated_reward: float = 0
        try:
            for j in range(duration_episode):
                value_state: torch.Tensor = critic(
                    1 / (env.nb_links - 1) * torch.Tensor(env.state.flatten())
                )
                action, action_clipped, norm_dist = sample_action(actor=actor, env=env)
                # Observe action and reward
                next_state, reward, _, _ = env.step(action_clipped.numpy().astype(int))
                cumulated_reward += reward
                value_next_state: torch.Tensor = critic(
                    torch.Tensor(next_state.flatten()))
                target: float = reward + GAMMA * value_next_state.detach()
                # Calculate losses
                critic_loss: torch.Tensor = critic_loss_fn(value_state, target)
                actor_loss: torch.Tensor = (
                        -norm_dist.log_prob(action).unsqueeze(0) * critic_loss.detach()
                )
                # Perform backpropagation
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()
                rewards_list.append(cumulated_reward)
                # Logs
                if j % 1000 == 0 and verbose == 1:
                    logging.info(
                        "Timestep: {}, Cumulated reward: {}".format(j, cumulated_reward)
                    )
                    logging.info(
                        "Minimal solution is : {} modems, {} groups".format(
                            env.nb_mod_min, env.nb_grps_min
                        )
                    )

        except ValueError:
            # Prevent the algorithm from stopping
            # if the loss of the actor becomes too
            # big
            pass

    return env.state_min, env.nb_mod_min, env.nb_grps_min
