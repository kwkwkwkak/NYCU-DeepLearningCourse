# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class cartDQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
           nn.Linear(4, 128),
           nn.ReLU(),
           nn.Linear(128, 128),
           nn.ReLU(),
           nn.Linear(128, num_actions)
        )       
    
    def forward(self, x):
        return self.network(x)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions, fcl_dim):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        linear_input_size = 3136 # precalcd

        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=fcl_dim)
        self.Alinear2 = nn.Linear(in_features=fcl_dim, out_features=num_actions)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=fcl_dim)
        self.Vlinear2 = nn.Linear(in_features=fcl_dim, out_features=1)
        
        ########## END OF YOUR CODE ##########

    # def conv2d_size_calc(self, w, h, kernel_size, stride):
    #     next_w = (w - (kernel_size - 1) - 1) // stride + 1
    #     next_h = (h - (kernel_size - 1) - 1) // stride + 1
    #     return next_w, next_h

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        Ax = nn.functional.relu(self.Alinear1(x))
        Ax = self.Alinear2(Ax) 

        Vx = nn.functional.relu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)

        return Vx + Ax - Ax.mean(dim=-1, keepdim=True)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized


    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha, beta_start, beta_annealing_steps):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment_per_step = (1.0 - beta_start) / beta_annealing_steps
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def __len__(self):
        return len(self.buffer)

    def add(self, transition):
        ########## YOUR CODE HERE (for Task 3) ########## 
        max_prio = np.max(self.priorities[:len(self.buffer)]) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ########## 

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        N = len(self.buffer)
        prios = self.priorities[:N]
        
        probs = prios / prios.sum()

        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights
        ########## END OF YOUR CODE (for Task 3) ########## 
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-6) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ########## 
    def beta_step(self):
        self.beta = min(1.0, self.beta + self.beta_increment_per_step)
        

class MultiStepPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, n_steps, gamma, alpha, beta_start, beta_annealing_steps):
        super().__init__(capacity, alpha, beta_start, beta_annealing_steps)
        self.n_steps = n_steps
        self.gamma = gamma
        self.history = deque(maxlen=n_steps)
    
    def add(self, transition):
        self.history.append(transition)

        if len(self.history) < self.n_steps:
            return
        
        #states, actions, rewards, next_states, terminated
        n_step_return = 0
        next_state, terminated = self.history[-1][3], self.history[-1][4]
        
        for k in range(self.n_steps):
            n_step_return += self.gamma ** k * self.history[k][2] # gamma^k * rt+k

            if self.history[k][4]: # if terminated, not truncated. If its truncated, the buffer will get cleared in the next step
                next_state = self.history[k][3]
                terminated = True
                break
            
        transition = self.history[0][0], self.history[0][1], n_step_return, next_state, terminated # use n step transition instead
    
        # same as super.add()
        super().add(transition)


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = cartDQN(self.num_actions).to(self.device) if env_name == "CartPole-v1" else DQN(self.num_actions, args.fcl_dim).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = cartDQN(self.num_actions).to(self.device) if env_name == "CartPole-v1" else DQN(self.num_actions, args.fcl_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0 if env_name == "CartPole-v1" else -21 # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.skip_preprocessing = env_name == "CartPole-v1"
        #self.memory = []
        self.memory_capacity = args.memory_size
        self.use_replay_buffer = True
        self.tau = 1e-3
        if self.use_replay_buffer:
            self.memory = MultiStepPrioritizedReplayBuffer(
                capacity=args.memory_size, 
                n_steps=3, 
                gamma=self.gamma, 
                alpha=0.5, 
                beta_start=0.4, 
                beta_annealing_steps=args.beta_annealing_steps
            )
            
            # self.memory = PrioritizedReplayBuffer(
            #     capacity=args.memory_size, 
            #     alpha=0.5, 
            #     beta_start=0.4, 
            #     beta_annealing_steps=1000000
            # )
        else:
            self.memory = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            
            if isinstance(self.memory, MultiStepPrioritizedReplayBuffer):
                self.memory.history.clear() # clear history to prevent cross-env returns
            
            state = obs if self.skip_preprocessing else self.preprocessor.reset(obs)
                
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = next_obs if self.skip_preprocessing else self.preprocessor.step(next_obs)

                if self.use_replay_buffer:
                    self.memory.add((state, action, reward, next_state, terminated)) # store terminated instead of done
                else:
                    self.memory.append((state, action, reward, next_state, terminated))
                    if len(self.memory) > self.memory_capacity:
                        self.memory.pop(0)    
                        
                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Memory Beta": self.memory.beta
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward >= self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        if self.skip_preprocessing:
            state = obs
        else:
            state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if self.skip_preprocessing:
                state = next_obs
            else:
                state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer

        if self.use_replay_buffer:
            mini_batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            mini_batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, terminateds = zip(*mini_batch)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        terminateds = torch.tensor(terminateds, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        # with torch.no_grad():
        #     target_q_values = rewards + self.gamma * (1 - terminateds) * self.target_net(next_states).max(1)[0]
        # DDQN
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            # when it is terminated, q_value should be zero. But q_value should still be calculated when truncated
            target_q_values = \
                rewards + self.gamma * (1 - terminateds) * self.target_net(next_states).gather(1, next_actions).squeeze(1)

        #loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        td_errors = q_values - target_q_values
        #
        squared_errors = td_errors ** 2
        # loss = nn.functional.smooth_l1_loss(q_values, target_values, reduction=None)
        
        if self.use_replay_buffer:
            loss = torch.mean(torch.tensor(weights, device=self.device) * squared_errors)
        else:
            loss = torch.mean(squared_errors)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10) # from duel-cnn paper
        self.optimizer.step()

        errors = td_errors.abs().detach().cpu().numpy()

        if self.use_replay_buffer:
            self.memory.update_priorities(indices, errors=errors)
            self.memory.beta_step()
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
           print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
        
        if self.env_count in [200000, 400000, 600000, 800000, 1000000]:
            model_path = os.path.join(self.save_dir, f"LAB5_110550022_task3_pong{self.env_count}.pt")
            torch.save(self.q_net.state_dict(), model_path)
            print(f"Saved model checkpoint to {model_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--save-dir", type=str, default="./results")
    # parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--memory-size", type=int, default=100000)
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--discount-factor", type=float, default=0.99)
    # parser.add_argument("--epsilon-start", type=float, default=1.0)
    # parser.add_argument("--epsilon-decay", type=float, default=0.999)
    # parser.add_argument("--epsilon-min", type=float, default=0.05)
    # parser.add_argument("--target-update-frequency", type=int, default=1000)
    # parser.add_argument("--replay-start-size", type=int, default=2000)
    # parser.add_argument("--max-episode-steps", type=int, default=500)
    # parser.add_argument("--train-per-step", type=int, default=1)
    # parser.add_argument("--beta-annealing-steps", type=int, default=200000)
    # parser.add_argument("--fcl-dim", type=int, default=128)
    
    # args = parser.parse_args()

    # wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    # agent = DQNAgent(args=args)
    # agent.run()
    
    #PONG
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999997)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=2000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=100000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--beta-annealing-steps", type=int, default=4000000)
    parser.add_argument("--fcl-dim", type=int, default=512)
    args = parser.parse_args()
    
    wandb.init(project="DLP-Lab5-DQN-Pong", name="pong-run", save_code=True)
    agent = DQNAgent(env_name="ALE/Pong-v5", args=args)
    agent.run(3000)
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--save-dir", type=str, default="./fast-train-base")
    # parser.add_argument("--wandb-run-name", type=str, default="pong-run")
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--memory-size", type=int, default=50000)
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--discount-factor", type=float, default=0.99)
    # parser.add_argument("--epsilon-start", type=float, default=0.6)
    # parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    # parser.add_argument("--epsilon-min", type=float, default=0.05)
    # parser.add_argument("--target-update-frequency", type=int, default=2000)
    # parser.add_argument("--replay-start-size", type=int, default=30000)
    # parser.add_argument("--max-episode-steps", type=int, default=100000)
    # parser.add_argument("--train-per-step", type=int, default=1)
    # parser.add_argument("--beta-annealing-steps", type=int, default=1000000)
    # parser.add_argument("--fcl-dim", type=int, default=256)
    # args = parser.parse_args()
    
    # wandb.init(project="DLP-Lab5-DQN-Pong", name="fast-train-base", save_code=True)
    # agent = DQNAgent(env_name="ALE/Pong-v5", args=args)
    # agent.run(500)