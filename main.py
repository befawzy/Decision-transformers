import numpy as np
import torch
import wandb
import argparse
import pickle
import time

from transformers import DecisionTransformerConfig, Trainer, TrainingArguments

from decision_transformer.model.decision_transformer import DecisionTransformer
from decision_transformer.data.data_collator import DecisionTransformerDataCollator
from decision_transformer.data.data_generator import generate_dataset

# this is the lr schedule used in the original paper. It might be changed later on
warmup_steps = variant['warmup_steps']


def main(arguments):

    device = arguments.get('device', 'cuda')
    log_to_wandb = arguments.get('log_to_wandb', False)
    # load dataset
    num_of_trajectories = arguments['num_of_trajectories']
    #load env parameters for data generation
    env_path= 'env/mean_env_params.pickle'
    #return data and input seed in the argument
    trajectories= generate_dataset(num_of_trajectories,env_path)
    # env_name = arguments['env']
    # #load generated data
    # dataset_path = f'data/{env_name}-dataset-{num_of_trajectories}.pkl'
    # with open(dataset_path, 'rb') as f:
    #     trajectories = pickle.load(f)

    collator = DecisionTransformerDataCollator(trajectories)
    config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim,
                                       max_length=collator.max_len,
                                       max_ep_len=arguments['max_ep_len'],
                                       hidden_size=arguments['embed_dim'],
                                       n_layer=arguments['n_layer'],
                                       n_head=arguments['n_head'],
                                       n_inner=4*arguments['embed_dim'],
                                       activation_function=arguments['activation_function'],
                                       n_positions=1024,
                                       resid_pdrop=arguments['dropout'],
                                       attn_pdrop=arguments['dropout']
                                       )
    model = DecisionTransformer(config)
    model = model.to(device=device)
    # optimizer = torch.optim.AdamW(
    #        DecisionTransformer.parameters(),
    #        lr=arguments['learning_rate'],
    #        weight_decay=arguments['weight_decay'],
    #    )

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lambda steps: min((steps+1)/warmup_steps, 1)
    # )
    training_args = TrainingArguments(
        output_dir="output/",
        remove_unused_columns=False,
        num_train_epochs=arguments['epochs'],
        per_device_train_batch_size=arguments['batch_size'],
        max_steps=arguments['max_iters'],
        learning_rate=arguments['learning_rate'],
        weight_decay=arguments['weight_decay'],
        warmup_ratio=arguments['warmup_ratio'],
        warmup_steps=arguments['warmup_steps'],
        lr_scheduler_type='linear',
        optim="adamw_torch",
        max_grad_norm=0.25,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trajectories,
        data_collator=collator,
    )
    trainer.train()
    #evaluation
    num_eval_episodes=arguments['num_eval_episodes']
    target_rewards=arguments['target_reward']
    def eval_fn(target):
        returns, lengths = [], []
        def fn(model):
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                                    state_dim=collator.state_dim,
                                    state_dim=collator.action_dim,
                                    model=model,
                                    max_ep_len=arguments['max_ep_len'],
                                    target_return=target,
                                    state_mean=collator.state_mean,
                                    state_std=collator.state_std,
                                    device=device,
                                )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target}_return_mean': np.mean(returns),
                f'target_{target}_return_std': np.std(returns),
                f'target_{target}_length_mean': np.mean(lengths),
                f'target_{target}_length_std': np.std(lengths),
            }
        return fn
    for target in (target_rewards): 
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, length = evaluate_episode_rtg(
                                state_dim=collator.state_dim,
                                state_dim=collator.action_dim,
                                model=model,
                                max_ep_len=arguments['max_ep_len'],
                                target_return=target,
                                state_mean=collator.state_mean,
                                state_std=collator.state_std,
                                device=device,
                            )
        model.eval()
        eval_fns=[eval_fn(target) for target in target_rewards]
        logs = dict()
        eval_start = time.time()
        for eval_fn in eval_fns:
            outputs = eval_fn(model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in diagnostics:
            logs[k] = diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='fractal')
    # medium, medium-replay, medium-expert, expert
    parser.add_argument('--num_of_trajectories', type=int, default=10000)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--max_ep_len', type=int, default=50)
    parser.add_argument('--activation_function', type=str, default='softmax')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--target_reward', type=list, default=[10,20])
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()

    main(variant=vars(args))
