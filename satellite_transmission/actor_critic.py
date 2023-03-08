"""Run the actor-crictic algorithm to solve the optimization problem."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from satellite_transmission.environment import SatelliteEnv
import satellite_transmission.config as config
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from termcolor import colored

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
RANDOM_SEED = 0

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
        return x.detach(), norm_dist


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


def run_actor_critic(links: list, nb_episodes: int, duration_episode: int, print_freq: int, log_freq: int, timeout: int, verbose: int, report: bool):
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
    if report:
        reward_per_time_step = []
        timesteps = []
        nb_modem_min_time_step = []
        nb_group_min_time_step = []
        episodes = []

    if verbose==2:
        print(
            "============================================================================================"
        )
        device = torch.device("cuda")
        if torch.cuda.is_available():
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set: cpu")
        print(
            "============================================================================================"
        )

        print(
            "============================================================================================"
        )
    env_name = "SatelliteEnv"
    ###################### report ######################
    if report:   
        results_dir = "actor-critic_Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        results_dir = results_dir + "/" + env_name + "/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        date = date.replace("/", "_")
        date = date.replace(":", "_")
        date = date.replace(" ", "_")

        results_dir = results_dir + "/" + date + "/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    #####################################################

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    log_dir = "actor_critic_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + "/" + env_name + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    date = date.replace("/", "_")
    date = date.replace(":", "_")
    date = date.replace(" ", "_")

    #### create new log file for each run
    log_f_name = log_dir + "/actor_critic_" + env_name + "_log_" + str(date) + ".csv"

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    if verbose==2:
        print("current logging run number for " + env_name + " : ", date)
        print("logging at : " + log_f_name)
        
    #####################################################


    env = SatelliteEnv(links)
    actor = ActorNetwork(obs_size=2 * len(links), action_size=3)
    critic = CriticNetwork(obs_size=2 * len(links))
    # Initiliaze loss
    critic_loss_fn = nn.MSELoss()
    # Initialize optimizers
    actor_optimizer = optim.Adam(params=actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(params=critic.parameters(), lr=LR_CRITIC)
    rewards_list = []

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ############# print all hyperparameters #############
    if verbose==2:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("max timesteps per episode : ", duration_episode)
        print("log frequency : " + str(log_freq) + " timesteps")
        print(
            "printing average reward over episodes in last : "
            + str(print_freq)
            + " timesteps"
        )
        print("timeout : " + str(timeout) + " timesteps")
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("state space dimension : ", state_dim)
        print("action space dimension : ", action_dim)
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("Initializing a discrete action space policy")
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("Actor-critic K epochs : ", nb_episodes)
        print("discount factor (gamma) : ", GAMMA)
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("optimizer learning rate actor : ", LR_ACTOR)
        print("optimizer learning rate critic : ", LR_CRITIC)
        
    if RANDOM_SEED:
        if verbose==2:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print("setting random seed to ", RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    #####################################################
    if verbose == 2:
        print(
            "============================================================================================"
        )

    start_time = datetime.now().replace(microsecond=0)
    if verbose==2:
        print("Started training at (GMT) : ", start_time)

        print(
            "============================================================================================"
        )
    for i_ep in range(nb_episodes):
        env.reset()
        cumulated_reward = 0
        try:
            for j in range(duration_episode):
                value_state = critic(
                    1 / (env.nb_links - 1) * torch.Tensor(env.state.flatten())
                )
                action, action_clipped, norm_dist = sample_action(actor=actor, env=env)
                # Observe action and reward
                next_state, reward, _, _ = env.step(action_clipped.numpy().astype(int))
                cumulated_reward += reward
                value_next_state = critic(torch.Tensor(next_state.flatten()))
                target = reward + GAMMA * value_next_state.detach()
                # Calculate losses
                critic_loss = critic_loss_fn(value_state, target)
                actor_loss = (
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
                elapsed_time = datetime.now().replace(microsecond=0) - start_time
                # Logs
                if j % print_freq == 0:
                    if verbose == 1:
                        logging.info(
                            "Episode: {}, Timestep: {}, Elapsed time: {}s".format(colored(i_ep,"blue"), colored(j,"blue"),colored(elapsed_time.seconds,"blue"))
                        )
                        logging.info(
                            "Cumulated reward: {}, Average reward: {}".format(
                                                                            colored(round(cumulated_reward,2), "green"),
                                                                            colored(round(np.mean(rewards_list),2), "green")
                                                                            )
                        )
                        logging.info(
                            "Minimal solution is : {} modems, {} groups\n".format(
                                colored(env.nb_mod_min,"yellow"), colored(env.nb_grps_min,"yellow")
                            )
                        )
                    elif verbose == 2:
                        print(
                        "--------------------------------------------------------------------------------------------"
                        )
                        print(
                            "Episode : {} \t\t Timestep : {} \t\t Elapsed time: {}s".format(
                                colored(i_ep,"blue"), colored(j,"blue"),colored(elapsed_time.seconds,"blue")
                            )
                        )
                        print(
                            "Cumulated reward: {}, Average reward: {}".format(
                                                                            colored(round(cumulated_reward,2), "green"),
                                                                            colored(round(np.mean(rewards_list),2), "green")
                                                                            )
                        )
                        print("Minimal solution is : {} modems, {} groups".format(
                                colored(env.nb_mod_min,"yellow"), colored(env.nb_grps_min,"yellow")
                            )
                        )
                        print(
                            "--------------------------------------------------------------------------------------------"
                        )

                if j % log_freq == 0:
                    log_f.write("{},{},{}\n".format(i_ep, j, cumulated_reward))
                    log_f.flush()
                
                if report:
                    reward_per_time_step.append(cumulated_reward)
                    nb_modem_min_time_step.append(env.nb_mod_min)
                    nb_group_min_time_step.append(env.nb_grps_min)
                    episodes.append(i_ep)
                    timesteps.append(j)


                if timeout != 0 and (datetime.now().replace(microsecond=0) - start_time).seconds > timeout:
                    if verbose == 1:
                        logging.info("Timeout reached, stopping the algorithm")
                    elif verbose == 2:
                        # print total training time
                        print(
                            "============================================================================================"
                        )
                        end_time = datetime.now().replace(microsecond=0)
                        print("Started training at (GMT) : ", start_time)
                        print("Finished training at (GMT) : ", end_time)
                        print("Total training time  : ", end_time - start_time)
                        print(
                            "============================================================================================"
                        )
                        print("============================================================================================")
                        print("TIMEOUT REACHED")
                        print("============================================================================================")
                    log_f.close()
                    if report:
                        df_time_step = pd.DataFrame(
                            {
                                "episode": episodes,
                                "timestep": timesteps,
                                "reward": reward_per_time_step,
                                "nb_modem_min": nb_modem_min_time_step,
                                "nb_group_min": nb_group_min_time_step,
                            }
                        )
                        df_time_step.to_csv(results_dir+"time_step_report.csv", index=False)
                    else:
                        results_dir = None
                    return env.state_min, env.nb_mod_min, env.nb_grps_min, results_dir
        except ValueError:
            # Prevent the algorithm from stopping
            # if the loss of the actor becomes too
            # big
            pass
    if verbose == 2:
        # print total training time
        print(
            "============================================================================================"
        )
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print(
            "============================================================================================"
        )
    elif verbose==1:
        logging.info("Training finished")

    log_f.close()

    if report:
        df_time_step = pd.DataFrame(
            {
                "episode": episodes,
                "timestep": timesteps,
                "reward": reward_per_time_step,
                "nb_modem_min": nb_modem_min_time_step,
                "nb_group_min": nb_group_min_time_step,
            }
        )
        df_time_step.to_csv(results_dir+"time_step_report.csv", index=False)
    else:
        results_dir = None

    return env.state_min, env.nb_mod_min, env.nb_grps_min, results_dir
