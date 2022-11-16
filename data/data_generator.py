from datasets import Dataset
import pyarrow as pa
import torch
from torch import nn
import pickle
from fractal_env_torch import FractalEnv
import numpy as np


def optimal_policy(prev_state: int) -> int:
    prev_state = int(prev_state)
    if prev_state == 0:
        action = 0
    elif prev_state == 1:
        action = 1
    elif prev_state == 2:
        action = 2
    elif prev_state == 3:
        action = 2
    else:
        raise ValueError
    return action


def generate_dataset(num_of_samples: int, env_path: str, epsilon: float, state_dim: int = 4, action_dim: int = 3) -> None:

    with open(env_path, "rb") as fp:
        env_params = pickle.load(fp)
    # initialize reward matrix
    reward_a_0 = - 0
    reward_a_R2 = - 50
    reward_a_A1 = - 2000
    reward_s_0 = - 100
    reward_s_1 = - 200
    reward_s_2 = - 1000
    reward_s_3 = - 8000

    reward_matrix = torch.as_tensor([
        [reward_a_0 + reward_s_0, reward_a_0 + reward_s_1,
            reward_a_0 + reward_s_2, reward_a_0 + reward_s_3],
        [reward_a_R2 + reward_s_0, reward_a_R2 + reward_s_1,
            reward_a_R2 + reward_s_2, reward_a_R2 + reward_s_3],
        [1*reward_a_A1 + reward_a_R2 + reward_s_0, 1.33*reward_a_A1 + reward_a_R2 + reward_s_1,
            1.66*reward_a_A1 + reward_a_R2 + reward_s_2, 2*reward_a_A1 + reward_a_R2 + reward_s_3]
    ])
    env = FractalEnv(reward_matrix=reward_matrix)
    # data generation
    data_ = {'observations': [],
             'actions': [],
             'rewards': [], 'dones': [], 'states': []}
    data_ = pa.Table.from_pydict(data_)
    dataset = Dataset(data_)

    for _ in range(50):

        obs, hidden_state = env.reset(env_params)
        obs_list, state_list, action_list, dones_list, reward_list = [], [], [], [], []
        for _ in range(50):
            if torch.rand(1) < 0.3:
                action = np.random.randint(3)
            else:
                action = optimal_policy(hidden_state)
            obs, hidden_state, reward, done, info = env.step(
                hidden_state, obs, action, env_params)
            obs_list.append(obs.tolist())
            state_list.append(int(hidden_state))
            action_list.append(action)
            reward_list.append(int(reward))
            dones_list.append(False)
        data_ = {'observations': obs_list,
                 'actions': nn.functional.one_hot(
                     torch.as_tensor(action_list), action_dim).tolist(),
                 'rewards': reward_list, 'dones': dones_list, 'states': nn.functional.one_hot(
                     torch.as_tensor(state_list), state_dim).tolist()}
        dataset = dataset.add_item(data_)
    return dataset


dataset = generate_dataset(1, 'env/mean_env_params.pickle', 0.3)
