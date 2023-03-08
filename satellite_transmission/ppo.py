"""Run the actor-crictic algorithm to solve the optimization problem."""
import os
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import logging
from satellite_transmission.environment import SatelliteEnv


logging.basicConfig(level=logging.INFO)

# =========================== Hyperparameters ===========================
################################## set device ##################################
# set device to cpu or cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

# NN layers
HIDDEN_SIZE = 256
# Discount factor
GAMMA = 0.99
# Learning rates
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001


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

        """dist = Categorical(action_probs)
        action = dist.sample()"""

        action_logprob = norm_dist.log_prob(action)
        state_val = self.critic(torch.transpose(state, 0, 1))[0]

        action_clipped = torch.clip(
            (self.state_dim - 1) * action,
            min=torch.zeros(3, dtype=torch.int),
            max=torch.Tensor(
                [self.state_dim - 1, self.state_dim - 1, self.state_dim - 1]
            ),
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
    def __init__(
        self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action_clipped, action, action_logprob, state_val = self.policy_old.act(
                state
            )

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
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

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


################################### Training ###################################
def run_ppo(links: list, nb_episodes: int, duration_episode: int, timeout: int, verbose: int):

    if verbose==1:
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
    env_name = "SatelliteEnv"

    max_ep_len = duration_episode  # max timesteps in one episode
    max_training_timesteps = (
        nb_episodes  # break training loop if timeteps > max_training_timesteps
    )

    print_freq = max_ep_len * 2  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1000)  # save model frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 5  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    if verbose==1:
        ("training environment name : " + env_name)
    env = SatelliteEnv(links)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.shape[0]

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + "/" + env_name + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + "/PPO_" + env_name + "_log_" + str(run_num) + ".csv"

    if verbose==1:
        print("current logging run number for " + env_name + " : ", run_num)
        print("logging at : " + log_f_name)
    #####################################################

    """################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################"""

    ############# print all hyperparameters #############
    if verbose==1:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("max training timesteps : ", max_training_timesteps)
        print("max timesteps per episode : ", max_ep_len)
        print("model saving frequency : " + str(save_model_freq) + " timesteps")
        print("log frequency : " + str(log_freq) + " timesteps")
        print(
            "printing average reward over episodes in last : "
            + str(print_freq)
            + " timesteps"
        )
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
        print("PPO K epochs : ", K_epochs)
        print("PPO epsilon clip : ", eps_clip)
        print("discount factor (gamma) : ", gamma)
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("optimizer learning rate actor : ", lr_actor)
        print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        if verbose==1:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################
    if verbose == 1:
        print(
            "============================================================================================"
        )

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(
        state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    if verbose==1:
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

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                if verbose==1:
                    print(
                        "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                            i_episode, time_step, print_avg_reward
                        )
                    )

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0 and verbose==1:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                print(f"{env.nb_grps_min = }")
                print(f"{env.nb_mod_min = }")
                # print("saving model at : " + checkpoint_path)
                # ppo_agent.save(checkpoint_path)
                # print("model saved")
                print(
                    "Elapsed Time  : ",
                    datetime.now().replace(microsecond=0) - start_time,
                )
                print(
                    "--------------------------------------------------------------------------------------------"
                )

            # break; if the episode is over
            elapsed_time = datetime.now().replace(microsecond=0) - start_time
            if done or (elapsed_time.seconds > timeout and timeout != 0):
                break
        
        if elapsed_time.seconds > timeout and timeout != 0:
            if verbose==1:
                print("============================================================================================")
                print("TIMEOUT REACHED")
                print("============================================================================================")
            break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    if verbose==1:
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

    return env.state_min, env.nb_grps_min, env.nb_mod_min
