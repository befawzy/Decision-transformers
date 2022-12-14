import numpy as np
import torch

def evaluate_rtg(
        env,
        env_params,
        state_dim,
        act_dim,
        model,
        max_ep_len=50,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='noise',
        model_name='pomdp'
    ):

    model.eval()
    model.to(device=device)

    if model_name=='pomdp':
        state, _ = env.reset(env_params)
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)
    elif model_name=='mdp':
        _, state = env.reset(env_params)
    else:
        raise ValueError     
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # the latest action and reward will be "padding"
    states = state.reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        obs, hidden_state, reward, done, _ = env.step(
                hidden_state, obs, action, env_params)
        if model_name=='POMDP':
            cur_state = obs.reshape(1, 1).to(device=device, dtype=torch.float32)
        
        elif model_name=='MDP':
            cur_state = nn.functional.one_hot(torch.as_tensor(hidden_state),4).to(device=device, dtype=torch.float32)
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length
