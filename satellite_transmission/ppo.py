"""Run the actor-crictic algorithm to solve the optimization problem."""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
logging.basicConfig(level=logging.INFO)

# =========================== Hyperparameters ===========================
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

# NN layers
HIDDEN_SIZE = 128
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
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, HIDDEN_SIZE),
                        nn.Tanh(),
                        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                        nn.Tanh(),
                        nn.Linear(HIDDEN_SIZE, 1)
                    )
                    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_probs = self.actor(torch.transpose(state, 0, 1))
        #print(action_probs[0], torch.Tensor.mean(action_probs[0]))
        
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
            max=torch.Tensor([self.state_dim - 1, self.state_dim - 1, self.state_dim - 1]),
        ).int()

        return action_clipped.detach(), action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = []
        for i in range(state.shape[0]):
            action_probs.append(self.actor(torch.transpose(state[i], 0, 1))[0])
        action_probs = torch.stack(action_probs)
       
        #dist = Categorical(action_probs)
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
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

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
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
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
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

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
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        min_nb_modem = np.unique(old_states[-1].numpy()[:,0]).shape[0]
        min_nb_group = np.unique(old_states[-1].numpy()[:,1]).shape[0]
        print("min_nb_modem: ", min_nb_modem, "min_nb_group: ", min_nb_group)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))