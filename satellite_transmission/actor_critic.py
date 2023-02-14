"""Run the actor-crictic algorithm to solve the optimization problem."""
import torch
import torch.nn as nn
import torch.optim as optim
from satellite_transmission.environment import SatelliteEnv
import satellite_transmission.config as config
import logging

logging.basicConfig(level=logging.INFO)
# Set the number of threads for Pytorch
torch.set_num_threads(config.TORCH_NB_THREADS)

# NN layers
HIDDEN_SIZE = 128
# Discount factor
GAMMA = 0.99
# Learning rates
LR_ACTOR = 0.00001
LR_CRITIC = 0.0001


class ActorNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        """Map a state to an action."""
        super().__init__()
        self.fc1_layer = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2_layer = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mu_out_layer = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size), nn.Sigmoid()
        )
        self.sigma_out_layer = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size), nn.Softplus()
        )

    def forward(self, x):
        x = self.fc1_layer(x)
        x = self.fc2_layer(x)
        mu = self.mu_out_layer(x)
        sigma_diag = self.sigma_out_layer(x)
        norm_dist = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag(sigma_diag)
        )
        x = norm_dist.sample()
        return x, norm_dist


class CriticNetwork(nn.Module):
    def __init__(self, obs_size):
        """Map a state to a quality-value."""
        super().__init__()
        self.fc1_layer = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2_layer = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.critic_out_layer = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1_layer(x)
        x = self.sigmoid(x)
        x = self.fc2_layer(x)
        x = self.sigmoid(x)
        x = self.critic_out_layer(x)
        return x


def sample_action(actor: ActorNetwork, env: SatelliteEnv):
    # Scale state
    state = 1 / (env.nb_links - 1) * torch.Tensor(env.state.flatten())
    action, norm_dist = actor(state)
    # Clip the action so that the values are in the action space
    action_clipped = torch.clip(
        (env.nb_links - 1) * action,
        min=torch.zeros(3, dtype=torch.int),
        max=torch.Tensor([env.nb_links - 1, env.nb_links - 1, env.nb_links - 1]),
    )
    return action, action_clipped, norm_dist


def run_actor_critic(links: list, nb_episodes: int, duration_episode: int):
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

    env = SatelliteEnv(links)
    actor = ActorNetwork(obs_size=2 * len(links), action_size=3)
    critic = CriticNetwork(obs_size=2 * len(links))
    # Initiliaze loss
    critic_loss_fn = nn.MSELoss()
    # Initialize optimizers
    actor_optimizer = optim.Adam(params=actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(params=critic.parameters(), lr=LR_CRITIC)
    rewards_list = []
    grp_mod_arrays = []
    for _ in range(nb_episodes):
        env.reset()
        cumulated_reward = 0
        try:
            for j in range(duration_episode):
                value_state = critic(
                    1 / (env.nb_links - 1) * torch.Tensor(env.state.flatten())
                )
                action, action_clipped, norm_dist = sample_action(actor=actor, env=env)
                # Observe action and reward
                next_state, reward, _, _ = env.step(
                    action_clipped.detach().numpy().astype(int)
                )
                cumulated_reward += reward
                value_next_state = critic(torch.Tensor(next_state.flatten()))
                target = reward + GAMMA * value_next_state
                # Calculate losses
                critic_loss = critic_loss_fn(target, value_state)
                actor_loss = (
                    -norm_dist.log_prob(action.detach()).unsqueeze(0)
                    * critic_loss.detach()
                )
                # Perform backpropagation
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()
                rewards_list.append(cumulated_reward)
                grp_mod_arrays.append(env.grp_mod_array)
                # Logs
                if j % 1000 == 0:
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
