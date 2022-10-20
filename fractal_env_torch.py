import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
from typing import Tuple, Union, Dict
from functools import partial
from torch.distributions import StudentT
import gym


class Discrete(object):
    """
    class for discrete spaces.

    """

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = torch.int32

    def sample(self, rng: torch.Generator('cpu')) -> torch.Tensor:
        """Sample random action uniformly from set of categorical choices."""
        return torch.randint(
            low=0, high=self.n, size=self.shape, generator=rng, dtype=self.dtype)

    def contains(self, x: torch.int) -> bool:
        """Check whether specific object is within space."""
        return True if (x < self.n and x >= 0) else False


class Box(object):
    """
    class for continuous spaces.

    """

    def __init__(
        self,
        low: float,
        high: float,
        shape: Tuple[int],
        dtype: torch.dtype = torch.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> torch.Tensor:
        """Sample random action uniformly from 1D continuous range."""
        return (self.high - self.low) * torch.rand(size=self.shape, dtype=self.dtype) + self.low

    def contains(self, x: torch.int) -> bool:
        """Check whether specific object is within space."""
        return True if torch.all(x >= self.low) and torch.all(x <= self.high) else False


class FractalEnv(gym.Env):
    """
    Jax-compatible implementation of fractal values environment.
    """

    def __init__(
        self,
        reward_matrix,
        seed=42
    ) -> None:
        self.reward_matrix = reward_matrix
        torch.manual_seed(seed)

    def step(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        action: Union[int, float],
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool, Dict]:
        """Performs step transitions in the environment."""
        obs_st, state_st, reward, done, info = self.step_env(
            state, obs, action, params
        )
        obs_re, state_re = self.reset_env(params)
        # Auto-reset environment based on termination
        state = state_re if done else state_st
        obs = obs_re if done else obs_st
        return obs, state, reward, done, info

    def reset(
        self, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(params)
        return obs, state

    def step_env(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        action: Union[int, float],
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool, Dict]:
        """Environment-specific step transition."""
        # sample reward
        reward = self.reward_matrix[action, state]
        # sample new state
        transition_matrices = params['p_transition']
        transition_probs = torch.as_tensor(transition_matrices[action, state])
        state = transition_probs.squeeze().multinomial(num_samples=1, replacement=True)
        if action == 0:
            obs = self.deterioration_process(state, obs, params)
        else:
            obs = self.repair_process(state, obs, action, params)

        return obs, state, reward, False, {}

    def reset_env(
        self, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Environment-specific reset."""
        # sample initial state
        init_probs = params['init_probs']
        state = torch.as_tensor(init_probs).multinomial(
            num_samples=1, replacement=True)
        # sample initial obs
        obs = self.init_process(state, params)
        return obs, state

    @property
    def name(self) -> str:
        """Environment name."""
        return "Fractal-values"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self) -> Discrete:
        """Action space of the environment."""
        return Discrete(3)

    def observation_space(self) -> Box:
        """Observation space of the environment."""
        return Box(-1e3, 0, shape=(1,), dtype=torch.float32)

    def state_space(self) -> Discrete:
        """State space of the environment."""
        return Discrete(4)

    def _deterioration_process(self, state, params):
        mu_d, sigma_d, nu_d = params['mu_d'][state], params[
            'sigma_d'][state], params['nu_d'][state]
        return StudentT(df=nu_d, loc=mu_d, scale=sigma_d).sample()

    def deterioration_process(self, state, obs, params):
        sample = self._deterioration_process(state, params)
        return sample+obs if sample < -obs else self.deterioration_process(state, obs, params)

    def _repair_process(self, state, obs, action, params):
        mu_r, sigma_r, nu_r, k = params['mu_r'][state], params['sigma_r'][
            state], params['nu_r'][state], params['k'][action-1]
        return StudentT(df=nu_r, loc=k*obs + mu_r, scale=sigma_r).sample()

    def repair_process(self, state, obs, action, params):
        sample = self._repair_process(state, obs, action, params)
        return sample if sample < 0.0 else self.repair_process(state, obs, action, params)

    def _init_process(self, state, params):
        mu_init, sigma_init, nu_init = params['mu_init'][state], params['sigma_init'][state], params['nu_init'][state]
        return StudentT(df=nu_init, loc=mu_init, scale=sigma_init).sample()

    def init_process(self, state, params):
        sample = self._init_process(state, params)
        return sample if sample < 0.0 else self.init_process(state, params)
