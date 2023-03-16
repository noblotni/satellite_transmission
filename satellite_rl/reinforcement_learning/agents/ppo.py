"""Run the actor-crictic algorithm to solve the optimization problem."""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from termcolor import colored

from satellite_rl.reinforcement_learning.environment import SatelliteEnv

logging.basicConfig(level=logging.INFO)

################################## set device ##################################
# set device to cpu or cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

################################## Hyperparameters ##################################
# NN layers
HIDDEN_SIZE = 128
# Discount factor
GAMMA = 0.99
# Learning rates
LR_ACTOR = 0.00001
LR_CRITIC = 0.0001
# Environment parameters
K_EPOCHS = 5  # update policy for K epochs in one PPO update
EPS_CLIP = 0.2  # clip parameter for PPO
RANDOM_SEED = 0  # set random seed if required (0 = no random seed)


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(torch.transpose(state, 0, 1))
        # print(action_probs[0], torch.Tensor.mean(action_probs[0]))

        norm_dist = torch.distributions.MultivariateNormal(
            loc=action_probs[0], covariance_matrix=torch.diag(action_probs[0])
        )
        action = norm_dist.sample()

        action_logprob = norm_dist.log_prob(action)
        state_val = self.critic(torch.transpose(state, 0, 1))[0]

        action_clipped = torch.clip(
            (self.state_dim - 1) * action,
            min=torch.zeros(3, dtype=torch.int),
            max=torch.Tensor([self.state_dim - 1, self.state_dim - 1, self.state_dim - 1]),
        ).int()

        return (
            action_clipped.detach(),
            action.detach(),
            action_logprob.detach(),
            state_val.detach(),
        )

    def evaluate(self, state, action):
        action_probs = []
        for i in range(state.shape[0]):
            action_probs.append(self.actor(torch.transpose(state[i], 0, 1))[0])
        action_probs = torch.stack(action_probs)

        # dist = Categorical(action_probs)
        norm_dist = torch.distributions.MultivariateNormal(
            loc=action_probs[0], covariance_matrix=torch.diag(action_probs[0])
        )
        action_logprobs = norm_dist.log_prob(action)
        dist_entropy = norm_dist.entropy()
        state_values = []
        for i in range(state.shape[0]):
            state_values.append(self.critic(torch.transpose(state[i], 0, 1))[0])
        state_values = torch.stack(state_values)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.K_epochs = K_EPOCHS

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": LR_ACTOR},
                {"params": self.policy.critic.parameters(), "lr": LR_CRITIC},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action_clipped, action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action_clipped

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )


def run_ppo(
    links: list,
    nb_episodes: int,
    duration_episode: int,
    print_freq: int,
    log_freq: int,
    timeout: int,
    verbose: int,
    report: bool,
    filename: Path,
    batch: bool,
    compare: bool,
):
    ### report variables ####
    if report:
        reward_per_time_step = []
        nb_modem_min_time_step = []
        nb_group_min_time_step = []
        episodes = []
        timesteps = []

    ### initialize environment hyperparameters ######
    if verbose == 2:
        print(
            "============================================================================================"
        )
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

    ####### initialize environment hyperparameters ######
    env_name = "SatelliteRL"

    max_ep_len = duration_episode  # max timesteps in one episode
    update_timestep = max_ep_len * 4  # update policy every n timesteps

    max_training_timesteps = nb_episodes  # break training loop if timeteps > max_training_timesteps

    #####################################################
    if verbose == 2:
        print("training environment name : " + env_name)
    env = SatelliteEnv(
        links,
        nb_groups_init=len(links),
        nb_modems_init=len(links),
        nb_modems_per_group=len(links),
    )

    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    action_dim = env.action_space.shape[0]

    ###################### report ######################
    if report:
        if not compare:
            results_dir = Path("satellite_rl/output/PPO_Results")
        else:
            results_dir = Path("satellite_rl/output/comparison")

        results_dir.mkdir(parents=True, exist_ok=True)

        results_dir /= env_name
        results_dir.mkdir(parents=True, exist_ok=True)

        if batch:
            results_dir /= "-".join(str(filename).split("-")[:-1])
        else:
            results_dir /= filename
        results_dir.mkdir(parents=True, exist_ok=True)
        if compare:
            results_dir /= "PPO"
            results_dir.mkdir(parents=True, exist_ok=True)
    #####################################################

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    log_dir = Path("satellite_rl/output/PPO_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_dir /= env_name
    log_dir.mkdir(parents=True, exist_ok=True)

    #### create new log file for each run
    log_f_name = log_dir / Path("PPO_" + str(env_name) + "_log_" + str(filename) + ".csv")

    if verbose == 2:
        print("current logging run number for " + env_name + " : ", filename)
        print("logging at : " + log_f_name)
    #####################################################

    ############# print all hyperparameters #############
    if verbose == 2:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("max training timesteps : ", max_training_timesteps)
        print("max timesteps per episode : ", max_ep_len)
        print("log frequency : " + str(log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
        print("timeout : " + str(timeout) + " seconds")
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
        print("PPO update frequency : " + str(update_timestep) + " timesteps")
        print("PPO K epochs : ", K_EPOCHS)
        print("PPO epsilon clip : ", EPS_CLIP)
        print("discount factor (gamma) : ", GAMMA)
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("optimizer learning rate actor : ", LR_ACTOR)
        print("optimizer learning rate critic : ", LR_CRITIC)

    if RANDOM_SEED:
        if verbose == 2:
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

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    if verbose == 2:
        print("Started training at (GMT) : ", start_time)

        print(
            "============================================================================================"
        )

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    i_episode = 0
    # training loop
    while i_episode <= nb_episodes:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            current_ep_reward += reward

            # update PPO agent
            if t % update_timestep == 0:
                ppo_agent.update()

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            # log in logging file
            if t % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write("{},{},{}\n".format(t, t, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if t % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                if verbose == 2:
                    print(
                        "--------------------------------------------------------------------------------------------"
                    )
                    print(
                        "Episode : {} \t\t Timestep : {} \t\t Elapsed time: {}s".format(
                            colored(i_episode, "blue"),
                            colored(t, "blue"),
                            colored(
                                (datetime.now().replace(microsecond=0) - start_time).seconds,
                                "blue",
                            ),
                        )
                    )
                    print(
                        "Cumulated reward: {}, Average reward: {}".format(
                            colored(round(print_running_reward, 2), "green"),
                            colored(round(print_avg_reward, 2), "green"),
                        )
                    )
                    print(
                        "Minimal solution is : {} modems, {} groups".format(
                            colored(env.nb_modems_min, "yellow"),
                            colored(env.nb_groups_min, "yellow"),
                        )
                    )
                    print(
                        "--------------------------------------------------------------------------------------------"
                    )
                elif verbose == 1:
                    logging.info(
                        "Episode: {}, Timestep: {}, Elapsed time: {}s".format(
                            colored(i_episode, "blue"),
                            colored(t, "blue"),
                            colored(
                                (datetime.now().replace(microsecond=0) - start_time).seconds,
                                "blue",
                            ),
                        )
                    )
                    logging.info(
                        "Cumulated reward: {}, Average reward: {}".format(
                            colored(round(print_running_reward, 2), "green"),
                            colored(round(print_avg_reward, 2), "green"),
                        )
                    )
                    logging.info(
                        "Minimal solution is : {} modems, {} groups\n".format(
                            colored(env.nb_modems_min, "yellow"),
                            colored(env.nb_groups_min, "yellow"),
                        )
                    )

                print_running_reward = 0
                print_running_episodes = 0

            elapsed_time = datetime.now().replace(microsecond=0) - start_time
            if report:
                reward_per_time_step.append(current_ep_reward)
                nb_modem_min_time_step.append(env.nb_modems_min)
                nb_group_min_time_step.append(env.nb_groups_min)
                episodes.append(i_episode)
                timesteps.append(t)

            # break; if the episode is over
            if done or (elapsed_time.seconds > timeout and timeout != 0):
                break

        if elapsed_time.seconds > timeout and timeout != 0:
            if verbose == 2:
                print(
                    "============================================================================================"
                )
                print("TIMEOUT REACHED")
                print(
                    "============================================================================================"
                )
            elif verbose == 1:
                logging.info("Timeout reached, stopping the algorithm")
            break

        i_episode += 1
    log_f.close()
    env.close()

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
        if batch:
            files = list(Path(results_dir).glob("report*.csv"))
            files.sort(key=lambda f: f.stat().st_mtime)
            if len(files) == 0:
                last_file_number = 0
            else:
                last_file = files[-1]
                last_file_number = int(last_file.name.split("_")[-1].split(".")[0]) + 1
            df_time_step.to_csv(Path(results_dir, f"report_{last_file_number}.csv"), index=False)
        else:
            df_time_step.to_csv(Path(results_dir, "report.csv"), index=False)
    return env.state_min, env.nb_groups_min, env.nb_modems_min
