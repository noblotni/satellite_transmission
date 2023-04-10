"""Run the actor-crictic algorithm to solve the optimization problem."""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from termcolor import colored

from satellite_rl import config
from satellite_rl.reinforcement_learning.environment import SatelliteEnv, greedy_initialisation

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
    """Map a state to an action."""

    def __init__(self, obs_size: int, action_size: int) -> None:
        super().__init__()
        self.fc1_layer: nn.Linear = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mu_out_layer: nn.Sequential = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size), nn.Sigmoid()
        )
        self.sigma_out_layer: nn.Sequential = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size), nn.Softplus()
        )

    def forward(
        self, input_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.distributions.MultivariateNormal]:
        """Perform calculations on the input to select an action.

        Args:
            input_x (torch.Tensor): scaled state

        Returns:
            sample_x (torch.Tensor): scaled action selected by the network
            norm_dist (torch.distributions.MultivariateNormal): normal probability distribution
                of the actions selected given the current state
        """

        input_x = self.fc1_layer(input_x)
        input_x = self.fc2_layer(input_x)
        output_mu: torch.Tensor = self.mu_out_layer(input_x)
        sigma_diag: torch.Tensor = self.sigma_out_layer(input_x)
        norm_dist = torch.distributions.MultivariateNormal(
            loc=output_mu, covariance_matrix=torch.diag(sigma_diag)
        )
        sample_x = norm_dist.sample()
        return sample_x.detach(), norm_dist


class CriticNetwork(nn.Module):
    """Map a state to a quality-value."""

    def __init__(self, obs_size: int) -> None:
        super().__init__()
        self.fc1_layer: nn.Linear = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.critic_out_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Perform calculations on the input to evaluate the action just performed.

        Args:
            input_x (torch.Tensor): scaled state

        Returns:
            output (torch.Tensor): quality value of the action
        """
        input_x = self.fc1_layer(input_x)
        input_x = self.sigmoid(input_x)
        input_x = self.fc2_layer(input_x)
        input_x = self.sigmoid(input_x)
        output = self.critic_out_layer(input_x)
        return output


class ACLogger:
    """Display every log messages related to Actor-Critic"""

    def __init__(self, filename: str, print_freq: int, log_freq: int):
        self.filename = filename
        self.print_freq = print_freq
        self.log_freq = log_freq
        self.log_file = self.create_log_file()

    def create_log_file(self):
        """Init the log file where to write the logs."""
        # log files for multiple runs are NOT overwritten
        log_dir = config.AC_LOG_DIR / config.ENV_NAME
        log_dir.mkdir(parents=True, exist_ok=True)

        # create new log file for each run
        log_file_path = log_dir / (
            "actor_critic_" + config.ENV_NAME + "_log_" + self.filename + ".csv"
        )

        # logging file
        log_file = open(log_file_path, "w+", encoding="utf-8")
        # Write headings
        log_file.write("episode,timestep,reward\n")
        logging.debug("current logging run number for " + config.ENV_NAME + " : " + self.filename)
        logging.debug("logging at : %s", log_file_path)
        return log_file

    def print_dash_line(self):
        """Print a line made of -."""
        dash_list = ["-"] * 50
        print("".join(dash_list))

    def print_equal_line(self):
        """Print a line made of =."""
        equal_list = ["="] * 50
        print("".join(equal_list))

    def print_configuration(
        self,
        duration_episode: int,
        timeout: int,
        state_dim: int,
        action_dim: int,
        nb_episodes: int,
    ):
        """Print actor-critic configuration."""
        self.print_dash_line()
        print("max timesteps per episode : ", duration_episode)
        print("log frequency : " + str(self.log_freq) + " timesteps")
        print(
            "printing average reward over episodes in last : " + str(self.print_freq) + " timesteps"
        )
        print("timeout : " + str(timeout) + " timesteps")
        self.print_dash_line()
        print("state space dimension : ", state_dim)
        print("action space dimension : ", action_dim)
        self.print_dash_line()
        print("Initializing a discrete action space policy")
        self.print_dash_line()
        print("Actor-critic K epochs : ", nb_episodes)
        print("discount factor (gamma) : ", GAMMA)
        self.print_dash_line()
        print("optimizer learning rate actor : ", LR_ACTOR)
        print("optimizer learning rate critic : ", LR_CRITIC)

    def display_start_training_event(self, start_time):
        """Display a message to signal the start of the training phase."""
        print("Started training at (GMT) : ", start_time)
        self.print_equal_line()

    def display_end_training_event(self, start_time, end_time):
        """Display a message to signal the end of the training phase."""
        self.print_equal_line()
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        self.print_equal_line()
        print("Training finished.")

    def display_timeout_event(self):
        """Display a signal to signal that the time is out."""
        self.print_equal_line()
        print("TIMEOUT REACHED")
        self.print_equal_line()

    def display_write_episode_logs(
        self, rewards_list: list, episode: int, timestep: int, elapsed_time, env: SatelliteEnv
    ):
        """Display and write logs during an episode."""
        if timestep % self.print_freq == 0:
            logging.info(
                "Episode: {}, Timestep: {}, Elapsed time: {}s".format(
                    colored(episode, "blue"),
                    colored(timestep, "blue"),
                    colored(elapsed_time.seconds, "blue"),
                )
            )
            logging.info(
                "Cumulated reward: {}, Average reward: {}".format(
                    colored(round(rewards_list[-1], 2), "green"),
                    colored(round(np.mean(rewards_list), 2), "green"),
                )
            )
            logging.info(
                "Minimal solution is : {} modems, {} groups\n".format(
                    colored(env.nb_modems_min, "yellow"),
                    colored(env.nb_groups_min, "yellow"),
                )
            )
        if timestep % self.log_freq == 0:
            self.log_file.write("{},{},{}\n".format(episode, timestep, rewards_list[-1]))
            self.log_file.flush()


class ACReporter:
    """Report data about Actor-Crictic algorithm."""

    def __init__(self, filename: str, batch: bool, compare: bool):
        self.filename = filename
        self.batch = batch
        self.compare = compare
        # List to keep track of variables
        self.reward_per_timestep = []
        self.timesteps = []
        self.nb_modems_min_time_step = []
        self.nb_groups_min_time_step = []
        self.episodes = []
        self.report_dir = self.create_report_directory()

    def create_report_directory(self):
        """Create the directory where the report is saved."""
        if not self.compare:
            report_dir = config.AC_REPORT_DIR / config.ENV_NAME
        else:
            report_dir = config.AC_COMPARISON_REPORT_DIR / config.ENV_NAME

        if not self.batch:
            report_dir = report_dir / self.filename
        else:
            report_dir /= "-".join(str(self.filename).split("-")[:-1])

        if self.compare:
            report_dir /= "actor-critic"

        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def update_data(self, reward, timestep, nb_modems_min, nb_groups_min, episode):
        """Append new data to the data stored."""
        self.reward_per_timestep.append(reward)
        self.timesteps.append(timestep)
        self.nb_groups_min_time_step.append(nb_groups_min)
        self.nb_modems_min_time_step.append(nb_modems_min)
        self.episodes.append(episode)

    def save_report_to_csv(self):
        """Save the report to a csv file."""
        df_time_step = pd.DataFrame(
            {
                "episode": self.episodes,
                "timestep": self.timesteps,
                "reward": self.reward_per_timestep,
                "nb_modem_min": self.nb_modems_min_time_step,
                "nb_group_min": self.nb_groups_min_time_step,
            }
        )
        df_time_step.to_csv(self.report_dir / "time_step_report.csv", index=False)


class ACAgent:
    """Actor-Critic agent."""

    def __init__(
        self,
        links: list,
        duration_episode: int,
        nb_episodes: int,
        timeout,
        logger: ACLogger,
        reporter: ACReporter,
        greedy_init: bool,
    ):
        self.links = links
        self.duration_episode = duration_episode
        self.nb_episodes = nb_episodes
        self.logger = logger
        self.reporter = reporter
        self.timeout = timeout
        if greedy_init:
            self.env: SatelliteEnv = greedy_initialisation(links=links)
        else:
            self.env: SatelliteEnv = SatelliteEnv(
                links=self.links, nb_groups_init=len(self.links), nb_modems_init=len(self.links)
            )
        self.actor: ActorNetwork = ActorNetwork(obs_size=2 * len(links), action_size=3)
        self.critic: CriticNetwork = CriticNetwork(obs_size=2 * len(links))
        # Initiliaze loss
        self.critic_loss_fn: nn.MSELoss = nn.MSELoss()
        # Initialize optimizers
        self.actor_optimizer: optim.Adam = optim.Adam(params=self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer: optim.Adam = optim.Adam(
            params=self.critic.parameters(), lr=LR_CRITIC
        )
        self.rewards_list: list = []

    def set_torch_device(self):
        """Set device on which pytorch runs."""
        self.logger.print_equal_line()
        device = torch.device("cuda")
        if torch.cuda.is_available():
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set: cpu")
        self.logger.print_equal_line()

    def set_random_seed(self):
        """Set a random seed if one is provided."""
        self.logger.print_dash_line()
        print("setting random seed to ", config.AC_RANDOM_SEED)
        torch.manual_seed(config.AC_RANDOM_SEED)
        self.env.seed(config.AC_RANDOM_SEED)
        np.random.seed(config.AC_RANDOM_SEED)
        self.logger.print_equal_line()

    def stop_at_timeout(self, start_time, end_time):
        """Stop the algorithm when timeout is reached."""
        self.logger.display_timeout_event()
        self.logger.display_end_training_event(start_time=start_time, end_time=end_time)
        report_dir = None
        if self.reporter:
            self.reporter.save_report_to_csv()
            report_dir = self.reporter.report_dir
        self.logger.log_file.close()
        return self.env.state_min, self.env.nb_modems_min, self.env.nb_groups_min, report_dir

    def run(self):
        """Start the agent."""
        self.set_torch_device()
        if config.AC_RANDOM_SEED:
            self.set_random_seed()
        self.logger.print_configuration(
            duration_episode=self.duration_episode,
            nb_episodes=self.nb_episodes,
            timeout=self.timeout,
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
        )
        start_time = datetime.now().replace(microsecond=0)
        self.logger.display_start_training_event(start_time)
        for i_ep in range(self.nb_episodes):
            self.env.reset()
            cumulated_reward: float = 0
            for j in range(self.duration_episode):
                value_state: torch.Tensor = self.critic(
                    scale_state(env=self.env, state=torch.Tensor(self.env.state))
                )
                action, action_clipped, norm_dist = sample_action(actor=self.actor, env=self.env)
                # Observe action and reward
                next_state, reward, _, _, _ = self.env.step(action_clipped.numpy().astype(int))
                cumulated_reward += reward
                value_next_state: torch.Tensor = self.critic(
                    scale_state(env=self.env, state=torch.Tensor(next_state))
                )
                target: float = reward + GAMMA * value_next_state.detach()
                # Calculate losses
                critic_loss: torch.Tensor = self.critic_loss_fn(value_state, target)
                actor_loss: torch.Tensor = (
                    -norm_dist.log_prob(action).unsqueeze(0) * critic_loss.detach()
                )
                # Perform backpropagation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.rewards_list.append(cumulated_reward)
                elapsed_time = datetime.now().replace(microsecond=0) - start_time
                self.logger.display_write_episode_logs(
                    rewards_list=self.rewards_list,
                    episode=i_ep,
                    timestep=j,
                    elapsed_time=elapsed_time,
                    env=self.env,
                )
                if self.reporter:
                    self.reporter.update_data(
                        reward=cumulated_reward,
                        timestep=j,
                        nb_modems_min=self.env.nb_modems_min,
                        nb_groups_min=self.env.nb_groups_min,
                        episode=i_ep,
                    )
                if (
                    self.timeout != 0
                    and (datetime.now().replace(microsecond=0) - start_time).seconds > self.timeout
                ):
                    end_time = datetime.now().replace(microsecond=0)
                    self.stop_at_timeout(start_time=start_time, end_time=end_time)
        end_time = datetime.now().replace(microsecond=0)
        self.logger.display_end_training_event(start_time=start_time, end_time=end_time)
        self.logger.log_file.close()
        report_dir = None
        if self.reporter:
            self.reporter.save_report_to_csv()
            report_dir = self.reporter.report_dir
        return (self.env.state_min, self.env.nb_modems_min, self.env.nb_groups_min, report_dir)


def sample_action(
    actor: ActorNetwork, env: SatelliteEnv
) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.MultivariateNormal]:
    """Select an action."""
    # Scale state
    state = torch.Tensor(env.state)
    state = scale_state(env, state)
    # Predict action
    action, norm_dist = actor(state)
    # Clip the action so that the values are in the action space
    action_clipped = torch.clip(
        torch.mul(torch.Tensor(env.high_action), action),
        min=torch.zeros(env.action_shape),
        max=torch.Tensor(env.high_action),
    )
    return action, action_clipped, norm_dist


def scale_state(env: SatelliteEnv, state: torch.Tensor):
    """Normalize the state."""
    high_state = torch.Tensor(env.high_obs)
    norm_tensor = torch.zeros_like(state)
    norm_tensor[:, 0] = 1 / high_state[0] * torch.ones(norm_tensor.shape[0])
    norm_tensor[:, 1] = 1 / high_state[1] * torch.ones(norm_tensor.shape[0])
    scaled_state = torch.flatten(torch.mul(norm_tensor, state))
    return scaled_state


def convert_verbosity_to_logging_level(verbosity: int):
    """Convert verbosity into a log level."""
    if verbosity == 1:
        return logging.INFO
    elif verbosity == 2:
        return logging.DEBUG
    else:
        return logging.WARNING


def run_actor_critic(
    links: list,
    nb_episodes: int,
    duration_episode: int,
    print_freq: int,
    log_freq: int,
    filename: str,
    timeout: int = 0,
    verbose: int = 1,
    report: bool = False,
    batch: bool = False,
    compare: bool = False,
    greedy_init: bool = False,
):
    """Run the actor-critic algorithm to solve the optimization problem.

    Args:
        links (list): list of satellite links to assign to modems and groups
        nb_episodes (int): number of episodes to run
        duration_episode (int): number of iterations of one episode
        print_freq (int): frequency of printing
        log_freq (int): frequency of the writing of logs in a logfile
        report (bool): indicate whether a report is made or not
        verbose (int): level of verbosity
        timeout (int): maximal duration of the training phase
        filename (str): filename for the log and report files
        batch (bool): indicate whether the algorithm runs in batch
            or not
        compare (bool): indicate whethher the algorithm is compared with
            PPO or not
        greedy_init (bool): initialize the environment greedily

    Returns:
        state_min (np.ndarray): the minimal state found
        nb_modems_min (int): the number of modems in the minimal state
        nb_groups_min (int): the number of groups in the minimal group
        results_dir (Path): Path to the results directory
    """
    verbose = convert_verbosity_to_logging_level(verbose)
    logging.basicConfig(level=verbose)
    logger = ACLogger(filename=filename, print_freq=print_freq, log_freq=log_freq)
    reporter = None
    if report:
        reporter = ACReporter(filename=filename, batch=batch, compare=compare)
    agent = ACAgent(
        links=links,
        duration_episode=duration_episode,
        nb_episodes=nb_episodes,
        logger=logger,
        reporter=reporter,
        timeout=timeout,
        greedy_init=greedy_init,
    )
    state_min, nb_modems_min, nb_groups_min, results_dir = agent.run()

    return state_min, nb_modems_min, nb_groups_min
