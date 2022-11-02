from datasets import Dataset
import pyarrow as pa
import torch
from torch import nn
import pickle
from decision_transformers.env.fractal_env_torch import FractalEnv

def generate_dataset(num_of_samples:int, env_path:str) -> None:

    with open(env_path, "rb") as fp:
        env_params = pickle.load(fp)
    #initialize reward matrix
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
    actions = {'Do-nothing': 0, 'Tamping': 1, 'Renewal': 2}
    env = FractalEnv(reward_matrix=reward_matrix)
    # data generation
    data_ = {'observations': [],
             'actions': [],
             'rewards': [], 'dones': [], 'states': []}
    data_ = pa.Table.from_pydict(data_)
    dataset = Dataset(data_)

    for i in range(num_of_samples):
        data_ = {'observations': [],
                 'actions': [],
                 'rewards': [], 'dones': [], 'states': []}
        obs, hidden_state = env.reset(env_params)
        obs_list = [obs.tolist()]
        state_list = [int(hidden_state)]
        action_list = [0]
        dones_list = [False]
        reward_list = [0]
        for _ in range(49):
            action= np.random.randint(3)
            obs, hidden_state, reward, done, info = env.step(
                hidden_state, obs, action, env_params)
            obs_list.append(obs.tolist())
            state_list.append(int(hidden_state))
            action_list.append(action)
            reward_list.append(int(reward))
            dones_list.append(False)
        data_['observations'].append(obs_list)
        data_['states'].append(state_list)
        data_['actions'].append(nn.functional.one_hot(
            torch.as_tensor(action_list), len(actions)).tolist())
        data_['rewards'].append(reward_list)
        data_['dones'].append(dones_list)
        dataset = dataset.add_item(data_)

    dataset_name = 'fractal_dataset'+str(num_of_samples)
    with open(f'{dataset_name}.pkl', 'wb') as f:
        pickle.dump(dataset, f)