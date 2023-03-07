from datasets import Dataset
import pyarrow as pa
import torch
from torch import nn
import pickle
from env.fractal_env_torch import FractalEnv
import numpy as np

# matrix defines the optimal policy from (https://arxiv.org/pdf/2212.07933.pdf) in the case of finite episodes of 50 timesteps
# rows represent finite timesteps, columns represent the states.
state_action_matrix = torch.zeros(50, 4)
state_action_matrix[:42] = torch.as_tensor([0, 1, 2, 2])
state_action_matrix[42:45] = torch.as_tensor([0, 1, 1, 2])
state_action_matrix[45] = torch.as_tensor([0, 1, 1, 1])
state_action_matrix[46:49] = torch.as_tensor([0, 0, 1, 1])
state_action_matrix[49] = torch.as_tensor([0, 0, 0, 0])
# matrix defines the optimal policy for the infinite horizon case from (https://arxiv.org/pdf/2212.07933.pdf)


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


def generate_dataset(num_of_samples: list, env_path: str, epsilon: list, max_ep_len: int, state_dim: int = 4, action_dim: int = 3) -> Dataset:

    with open(env_path, "rb") as fp:
        env_params = pickle.load(fp)
    env = FractalEnv(seed=40)
    # data generation
    data_ = {'observations': [],
             'actions': [],
             'rewards': [], 'dones': [], 'states': []}
    data_ = pa.Table.from_pydict(data_)
    dataset = Dataset(data_)
    if type(epsilon) is not list:
        epsilon = [epsilon]
    if type(num_of_samples) is not list:
        num_of_samples = [num_of_samples]
    for i in range(len(epsilon)):

        for _ in range(int(num_of_samples[i])):
            obs, hidden_state = env.reset(env_params)
            obs_list = [obs.tolist()]
            state_list = [int(hidden_state)]
            action_list = []
            dones_list = []
            reward_list = []
            for j in range(max_ep_len):
                if torch.rand(1) < epsilon[i]:
                    action = np.random.randint(3)
                else:
                    action = int(state_action_matrix[j, hidden_state])
                # this if statement could be used in case of
                # generating data in the infinite horizon
                # if torch.rand(1) < epsilon[i]:
                #     action = np.random.randint(3)
                # else:
                #     action = optimal_policy(int(hidden_state))
                obs, hidden_state, reward, _, _ = env.step(
                    hidden_state, obs, action, env_params)

                action_list.append(action)
                reward_list.append(int(reward))
                dones_list.append(False)
                if j != max_ep_len-1:
                    obs_list.append(obs.tolist())
                    state_list.append(int(hidden_state))
            data_ = {'observations': obs_list,
                     'actions': nn.functional.one_hot(
                         torch.as_tensor(action_list), action_dim).tolist(),
                     'rewards': reward_list, 'dones': dones_list, 'states': nn.functional.one_hot(
                         torch.as_tensor(state_list), state_dim).tolist()}
            dataset = dataset.add_item(data_)
    return dataset
