#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple
import datetime
import pickle
from collections import deque

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return input * torch.tanh(F.softplus(input))
    
class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            Mish(),
            nn.Linear(64, 64),
            Mish(),
            nn.Linear(64, out_dim)
        )
        
        self.logstds = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, X):
        means = self.model(X)
        stds = self.logstds.clamp(-20, 0).exp()
        
        dist = torch.distributions.Normal(means, stds)
        return dist.sample(), dist

class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            Mish(),
            nn.Linear(64, 64),
            Mish(),
            nn.Linear(64, 1),
        )
    
    def forward(self, X):
        return self.model(X)


class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.logging = args.wandb
        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.memory = []
        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        return selected_action.cpu().detach().numpy(), dist.log_prob(selected_action), dist.entropy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        # state, log_prob, next_state, reward, done, action = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        actions, states, next_states, rewards, dones, log_probs, entropies = zip(*self.memory)
        
        states = torch.stack(states)
        actions = torch.stack(actions).unsqueeze(-1)
        next_states = torch.stack(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # print(states.shape, actions.shape, next_states.shape, rewards.shape, dones.shape)
        values = self.critic(states)
        next_values = self.critic(next_states)

        targets = rewards + self.gamma * next_values * (1 - dones)
        value_loss = F.mse_loss(values, targets.detach())

        advantages = (targets - values).detach()
        entropy_loss = entropies.mean()
        
        policy_loss = (
            -log_probs * advantages
        ).mean() - self.entropy_weight * entropy_loss


        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()


        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def to_t(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        score = 0
        done = False
        ep = 0
        step_per_ep = 200
        mini_batch_size = 16
        pbar = tqdm(range(1, self.num_episodes * step_per_ep // mini_batch_size + 1))
        state, _ = self.env.reset(seed=self.seed)
        
        for _ in pbar:
            self.memory.clear()

            for i in range(mini_batch_size):
                action, log_prob, entropy = self.select_action(state)
                clamped_action = np.clip(action, -2, 2)

                next_state, reward, done = self.step(clamped_action)
                self.memory.append(
                    [
                        self.to_t(action),
                        self.to_t(state),
                        self.to_t(next_state),
                        reward,
                        done,
                        log_prob,
                        entropy
                    ]
                )
                
                state = next_state
                score += reward
                step_count += 1

                if done:
                    ep += 1
                    state, _ = self.env.reset(seed=self.seed)

                    if ep % 10 == 0:
                        pbar.set_description(f"Episode {ep}: Total Reward = {score}")
                    if self.logging:
                        wandb.log({"episode": ep, "return": score})
                    score = 0

            actor_loss, critic_loss, entropy_loss = self.update_model()
            
            if step_count % 10000 == 0:
                self.save_models(ep=step_count)
                
            if self.logging:
                wandb.log(
                    {
                        "step": step_count,
                        "actor loss": actor_loss,
                        "critic loss": critic_loss,
                        "entropy loss": entropy_loss
                    }
                )
    @torch.no_grad
    def test(self, video_folder: str=None):
        """Test the agent."""
        self.is_test = True
        self.env._update_running_mean = False

        tmp_env = self.env
        if video_folder:
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action, _, _ = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env
        
        return score
    
    def save_models(self, save_dir: str = "a2c", ep: int = None):
        os.makedirs(save_dir, exist_ok=True)

        actor_path = os.path.join(
            save_dir, f"actor{'_' + str(ep) if ep is not None else ''}.pt"
        )
        critic_path = os.path.join(
            save_dir, f"critic{'_' + str(ep) if ep is not None else ''}.pt"
        )
        env_path = os.path.join(
            save_dir, f"env{'_' + str(ep) if ep is not None else ''}.pkl"
        )
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        with open(env_path, "wb") as f:
            pickle.dump({
                "mean": env.obs_rms.mean,
                "var": env.obs_rms.var,
                "count": env.obs_rms.count,
            }, f)
            
        print(f"Models saved to {actor_path} and {critic_path}")
    
    def load_models(self, save_dir: str = "a2c", ep: int = None):
        actor_path = os.path.join(
            save_dir, f"actor{'_' + str(ep) if ep is not None else ''}.pt"
        )
        critic_path = os.path.join(
            save_dir, f"critic{'_' + str(ep) if ep is not None else ''}.pt"
        )
        env_path = os.path.join(
            save_dir, f"env{'_' + str(ep) if ep is not None else ''}.pkl"
        )
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        
        with open(env_path, "rb") as f:
            stats = pickle.load(f)

        env.obs_rms.mean = stats["mean"]
        env.obs_rms.var = stats["var"]
        env.obs_rms.count = stats["count"]
    
    def load_models_(self, save_dir: str = "a2c"):
        actor_path = os.path.join(
            save_dir, f"actor.pt"
        )
        critic_path = os.path.join(
            save_dir, f"critic.pt"
        )
        env_path = os.path.join(
            save_dir, f"env.pkl"
        )
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        
        with open(env_path, "rb") as f:
            stats = pickle.load(f)

        env.obs_rms.mean = stats["mean"]
        env.obs_rms.var = stats["var"]
        env.obs_rms.count = stats["count"]


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=4e-4)
    parser.add_argument("--critic-lr", type=float, default=4e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--entropy-weight", type=int, default=1e-2
    )  # entropy can be disabled by setting this to 0
    parser.add_argument(
        "--wandb", action="store_true"
    )  # entropy can be disabled by setting this to 0
    args = parser.parse_args()

    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = gym.wrappers.NormalizeObservation(env)
    seed = int(datetime.datetime.now().timestamp())

    if args.wandb:
        wandb.init(
            project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name + str(seed), save_code=True
        )

    agent = A2CAgent(env, args)
    #agent.train()
    #agent.load_models(save_dir="a2c_final", ep=150000)
    agent.load_models_(save_dir="../Lab7_110550022_task1_a2c_pendulum/")
    # the model is from step 150k.
    seeds = [1, 2, 6, 7, 8, 12, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 33, 35]
    for seed in seeds:
        agent.seed = seed
        agent.test()