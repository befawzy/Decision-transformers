import torch
import pickle
from torch import nn
from fractal_env_torch import FractalEnv
from transformers import DecisionTransformerConfig
from decision_transformer import DecisionTransformer
import numpy as np

def load_model(input_path):
    config = DecisionTransformerConfig.from_json_file(input_path + '/config.json')   
    model=DecisionTransformer(config)  
    model.load_state_dict(torch.load(input_path+'/pt'))
    return model
 


def evaluate(model,target_return, env_path,seed, model_name= 'MDP', device='cpu'):
    model.to(device=device)
    TARGET_RETURN = target_return/50
    episode_return, episode_length = 0, 0
    with open(env_path, "rb") as fp:
        env_params = pickle.load(fp)
    env = FractalEnv(seed= seed)
    obs, hidden_state = env.reset(env_params)

    device='cpu'
    target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)

    if model_name=='POMDP':
        states = obs.reshape(1, 1).to(device=device, dtype=torch.float32)

    elif model_name=='MDP':
        states = nn.functional.one_hot(torch.as_tensor(hidden_state),4).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, 3), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    max_ep_len=50
    states_list=[]
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, 3), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states,
            actions,
            rewards,
            target_return,
            timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        obs, hidden_state, reward, done, _ = env.step(
                    hidden_state, obs, action, env_params)
        if model_name=='POMDP':
            cur_state = obs.reshape(1, 1).to(device=device, dtype=torch.float32)

        elif model_name=='MDP':
            cur_state = nn.functional.one_hot(torch.as_tensor(hidden_state),4).to(device=device)     

        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        pred_return = target_return[0, -1] - reward/50
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1
        states_list.append(int(hidden_state))

        if done:
            break
    actions_list = [int(actions[i,0]) for i in range(actions.shape[0])]
    return torch.as_tensor(actions_list),torch.as_tensor(states_list),episode_return
