#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym
import os
import datetime
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

# def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
#     """Init uniform parameters on the single layer."""
#     layer.weight.data.uniform_(-init_w, init_w)
#     layer.bias.data.uniform_(-init_w, init_w)

#     return layer

# def init_weights(block: nn.Sequential):
#     for layer in block:
#         if isinstance(layer, nn.Linear):
#             init_layer_uniform(layer)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 2,
        hidden_dim=64,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.mean_layers = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, out_dim), std=0.01),
        )

        self.logstds = nn.Parameter(torch.zeros(1, out_dim))
        self.logstds_min = log_std_min
        self.logstds_max = log_std_max

        #init_weights(self.mean_layers)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        mean = self.mean_layers(state)
        std = self.logstds.clamp(self.logstds_min, self.logstds_max)
        std = std.expand_as(mean).exp()

        dist = Normal(mean, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.model = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        #init_weights(self.model)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        value = self.model(state)
        #############################

        return value
    
def compute_gae(
    next_value: torch.tensor, rewards: list, masks: list, values: list, gamma: float, tau: float
) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = 0
    gae_returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
# def ppo_iter(
#     update_epoch: int,
#     mini_batch_size: int,
#     states: torch.Tensor,
#     actions: torch.Tensor,
#     values: torch.Tensor,
#     log_probs: torch.Tensor,
#     returns: torch.Tensor,
#     advantages: torch.Tensor,
# ):
#     """Get mini-batches."""
#     batch_size = states.size(0)
#     for _ in range(update_epoch):
#         for _ in range(batch_size // mini_batch_size):
#             rand_ids = np.random.choice(batch_size, mini_batch_size)
#             yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
#                 rand_ids
#             ], returns[rand_ids], advantages[rand_ids]

# ppo iter using shuffling
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    batch_size = states.size(0)
    for _ in range(update_epoch):
        # Random permutation of all indices (no replacement)
        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            batch_ids = indices[start:end]

            yield (
                states[batch_ids],
                actions[batch_ids],
                values[batch_ids],
                log_probs[batch_ids],
                returns[batch_ids],
                advantages[batch_ids],
            )

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        self.wandblog = args.wandb

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action).sum(1))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)
        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions).reshape(-1, self.action_dim)
        returns = torch.cat(returns).detach().reshape(-1)
        values = torch.cat(self.values).detach().reshape(-1)
        log_probs = torch.cat(self.log_probs).detach().reshape(-1)
        advantages = returns - values

        actor_losses, critic_losses, entropy_losses = [], [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action).sum(1)
            ratio = (log_prob - old_log_prob).exp()
            value = self.critic(state).view(-1)
            
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) # mini batch normalization
            # actor_loss
            ############TODO#############

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            entropy_loss = dist.entropy().sum(1).mean()

            actor_loss = (
                torch.max(pg_loss1, pg_loss2).mean()
                - self.entropy_weight * entropy_loss
            )

            #############################

            # critic_loss
            ############TODO#############
            critic_loss = F.mse_loss(value, return_)

            #############################

            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        entropy_loss = sum(entropy_losses) / len(entropy_losses)

        return actor_loss, critic_loss, entropy_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False
        
        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses, entropy_losses = [], [], []
        scores = []
        score = 0
        episode_count = 0
        pbar = tqdm(range(1, self.num_episodes))
        save_counter = 0
        for ep in pbar:
            
            score = 0
            print("\n")
            for _ in range(self.rollout_len):
                self.total_step += 1
                if self.total_step in [1e6, 1.5e6, 2e6, 2.5e6, 3e6]:
                    self.save_models(ep=self.total_step)
                    
                action = self.select_action(state)
                action = action.reshape(self.action_dim,)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    pbar.set_description(f"Episode {episode_count}: Total Reward = {score}")
                    if self.wandblog:
                        wandb.log({
                            "step": self.total_step,
                            "score": score
                        }) 
                    score = 0

            actor_loss, critic_loss, entropy_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_losses.append(entropy_loss)
            if self.wandblog:
                wandb.log({
                    "step": self.total_step,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy_loss": entropy_loss
                }) 
            
            if (save_counter + 1) % 50 == 0:
                self.save_models(ep=episode_count)
            save_counter += 1
            
        # termination
        self.env.close()

    @torch.no_grad()
    def test(self, video_folder: str = None):
        """Test the agent."""
        self.is_test = True
        #self.env.training = False
        self.env._update_running_mean = False

        tmp_env = self.env
        if video_folder is not None:
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            action = action.reshape(self.action_dim,)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()
        
        self.env = tmp_env
        
        return score
    
    def save_models(self, save_dir: str = "ppo_walker_models", ep: int = None):
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
    
    def load_models(self, save_dir: str = "ppo_walker_models", ep: int = None):
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

def make_env_with_wrappers():
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    env = gym.wrappers.NormalizeObservation(env)
    
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--entropy-weight", type=int, default=1e-2)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2048)  
    parser.add_argument("--update-epoch", type=float, default=10)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
 
    # environment
    env = make_env_with_wrappers()
    
    seed = int(datetime.datetime.now().timestamp())
    # print("torch, np, rand seed:", seed)
    # random.seed(seed)
    # np.random.seed(seed + 1)
    # seed_torch(seed + 2)
    if args.wandb:
        wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name + str(seed), save_code=True)
    
    agent = PPOAgent(env, args)
    #agent.load_models("ppo_walker_models_4k_final", ep=1000000)
    agent.load_models_(save_dir="../Lab7_110550022_task3_ppo_1p5m/")
    #agent.train()
    seeds = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for seed in seeds:
        agent.seed = seed
        agent.test()
        