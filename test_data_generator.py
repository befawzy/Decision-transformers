from datasets import Dataset
import pyarrow as pa
import torch
from torch import nn
import pickle
from fractal_env_torch import FractalEnv
import numpy as np

#matrix defines the optimal policy in the case of finite episodes .rows represent finite timesteps, columns represent the states. 
state_action_matrix=torch.zeros(50,4)
state_action_matrix[:42]=torch.as_tensor([0, 1, 2, 2])
state_action_matrix[42:45] =torch.as_tensor([0, 1, 1, 2])
state_action_matrix[45] =torch.as_tensor([0, 1, 1, 1])
state_action_matrix[46:49] =torch.as_tensor([0, 0, 1, 1])
state_action_matrix[49] =torch.as_tensor([0, 0, 0, 0])

#example: if we are at timestep 0 and want to select the optimal action given that the previous state is 2 we do the following:
#int(state_action_matrix[0,2])

def generate_dataset(num_of_samples: int, env_path: str, epsilon: float, state_dim: int = 4, action_dim: int = 3) -> Dataset:

    with open(env_path, "rb") as fp:
        env_params = pickle.load(fp)
    # initialize reward matrix
    env = FractalEnv()
    # data generation
    data_ = {'observations': [],
             'actions': [],
             'rewards': [], 'dones': [], 'states': []}
    data_ = pa.Table.from_pydict(data_)
    dataset = Dataset(data_)
    if type(epsilon) is not list:
        epsilon=[epsilon]
    for eps in epsilon:

        for _ in range(int(num_of_samples)):

            obs, hidden_state = env.reset(env_params)
            obs_list = []
            state_list = []
            action_list = []
            dones_list = []
            reward_list = []
            for i in range(50):
                if torch.rand(1) < eps:
                    action = np.random.randint(3)
                else:
                    action = int(state_action_matrix[i,hidden_state])
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


# env_path='env/mean_env_params.pickle'
# model_name='MDP'
# n_trajs=500
# epsilon=0
# data_0=generate_dataset(n_trajs,'env/mean_env_params.pickle',0)
# data_0_1=generate_dataset(n_trajs,'env/mean_env_params.pickle',0.1)
# data_0_3=generate_dataset(n_trajs,'env/mean_env_params.pickle',0.3)
# data_0_5=generate_dataset(n_trajs,'env/mean_env_params.pickle',0.5)
# import pandas as pd
# concat= pd.concat([data_0.to_pandas(), data_0_1.to_pandas(), data_0_3.to_pandas()])
# concat['returns']=concat['rewards'].apply(np.sum)
# import matplotlib.pyplot as plt 
# bins= np.linspace(np.min(concat['returns'])-1000, np.max(concat['returns'])+1000, 100)
# plt.hist(np.stack(concat['returns']), bins)
# plt.hist(data_0.to_pandas()['rewards'].apply(np.sum), bins=bins)
# plt.hist(data_0_1.to_pandas()['rewards'].apply(np.sum), bins=bins)
# plt.hist(data_0_3.to_pandas()['rewards'].apply(np.sum), bins=bins)
# plt.hist(data_0_5.to_pandas()['rewards'].apply(np.sum), bins=bins)
# states=data_0_5.to_pandas()['states'].apply(lambda x : [np.argmax(i) for i in x])
# for j in range(20):
#     plt.figure()
#     plt.plot(np.arange(50),states[j])


