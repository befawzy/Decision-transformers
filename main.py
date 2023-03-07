import numpy as np
import torch
import argparse
import pickle
import torch
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from model.decision_transformer import DecisionTransformer
from data.data_collator import DecisionTransformerDataCollator
from data.data_generator import generate_dataset
from evaluation.evaluate import load_model, evaluate
import seaborn as sns


def evaluation(target_vals, num_evals, model, policy_target_val_ind, model_name, env_path):
    returns_mean = []
    returns_median = []
    returns_std = []
    seeds = np.random.SeedSequence()
    ret_tens = torch.zeros((target_vals.shape[0], num_evals))
    act_tens, state_tens = torch.zeros((num_evals, 50, target_vals.shape[0])), torch.zeros(
        (num_evals, 51, target_vals.shape[0]))
    for j in range(target_vals.shape[0]):
        seedseq = seeds.generate_state(num_evals)
        for i in range(num_evals):
            model.eval()
            act_, state_, ret_tens[j, i] = evaluate(
                model, target_vals[j], model_name=model_name, env_path=env_path, seed=seedseq[i])
            state_tens[i, :, j] = state_.argmax(dim=1)
            act_tens[i, :, j] = act_.argmax(dim=1)
        returns_mean.append(int(torch.mean(ret_tens[j, :])))
        returns_median.append(torch.median(ret_tens[j, :]))
        returns_std.append(int(torch.std(ret_tens[j, :])))
    # plot the achieved return versus target return
    sns.set()
    sns.set_style("darkgrid")
    plt.rcParams["font.size"] = "11"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.plot(target_vals, returns_mean, color='b',
             label='mean achieved return', marker='.', markersize=5)
    plt.plot(target_vals, target_vals, color='r',
             label='desired return', marker='.', markersize=5)
    plt.fill_between(target_vals, np.array(returns_mean) + np.array(returns_std),
                     np.array(returns_mean) - np.array(returns_std), color='b', alpha=0.3)
    plt.legend()
    ax = plt.gca()
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False)
    print('mean achieved return for ')
    print(returns_mean)
    print('std of the achieved return')
    print(returns_std)
    # policy estimation
    actions_tens = np.empty((4, 50))
    actions_percent = np.empty((4, 50))

    for t in range(50):
        # get most chosen action per each state
        for state in np.arange(4):
            action_count = torch.bincount(torch.tensor(
                act_tens[:, t, policy_target_val_ind][state_tens[:, t, policy_target_val_ind] == state], dtype=int))
            action_choice = torch.argmax(action_count)
            actions_tens[state, t] = action_choice.tolist()
            actions_percent[state, t] = (torch.round(
                action_count[action_choice]/torch.sum(action_count), decimals=2)).tolist()
    for state in np.arange(4):
        print(f'printing chosen actions for state {state}')
        print(actions_tens[state, :])
        print(f'printing frequency of the chosen actions for state {state}')
        print(actions_percent[state, :]*100)


def experiment(variant):
    device = variant['device']
    epsilon = variant['epsilon']
    model_name = variant['model_name']
    env_path = variant['env_path']
    K = variant['context_length']
    n_trajs = variant['num_of_trajectories']
    epochs = variant['epochs']
    n_layers = variant['n_layers']
    n_heads = variant['n_heads']
    dataset_path = f'data/dataset-{n_trajs}-{epsilon}.pkl'
    data = generate_dataset(
        n_trajs, env_path, epsilon, max_ep_len=50)
    if variant['load_data']:
        with open(dataset_path, 'wb') as f:
            pickle.dump(data, f)

    eval_data = generate_dataset(10, env_path, epsilon, max_ep_len=50)
    collator = DecisionTransformerDataCollator(data, model_name=model_name)
    config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim,
                                       max_length=K,
                                       max_ep_len=50,
                                       hidden_size=variant['embed_dim'],
                                       n_layer=variant['embed_dim'],
                                       n_head=variant['n_heads'],
                                       n_inner=4*128,
                                       n_positions=1024,
                                       resid_pdrop=variant['dropout'],
                                       attn_pdrop=variant['dropout']
                                       )
    model = DecisionTransformer(config)
    output_dir = f'output/{model_name}/samples_{n_trajs}/epsilon_{epsilon}/epochs_{epochs}/max_len{K}/n_layer{n_layers}/n_heads{n_heads}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
        lr_scheduler_type='linear',
        optim="adamw_torch",
        max_grad_norm=0.25,
        evaluation_strategy='epoch',
        logging_dir=output_dir+'/logs',
        logging_strategy='epoch')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        eval_dataset=eval_data,
        data_collator=collator,
    )
    trainer.train()
    if not variant['load_model']:
        torch.save(model.state_dict(), output_dir+'/pt')
        config.save_pretrained(output_dir)
    target_vals = variant['target_vals']
    if type(target_vals) is list:
        target_vals = np.array(target_vals)

    num_evals = variant['num_evals']
    policy_target_val = variant['target_val_for_policy']
    if not np.any(target_vals == policy_target_val):
        raise ValueError
    policy_target_val_ind = int(np.argwhere(target_vals == policy_target_val))
    evaluation(target_vals=target_vals, num_evals=num_evals, model=model,
               policy_target_val_ind=policy_target_val_ind, model_name=model_name, env_path=env_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', type=str,
                        default='env/mean_env_params.pickle')

    parser.add_argument('--num_of_trajectories', type=int, default=50)
    parser.add_argument('--context_length', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--max_ep_len', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--num_evals', type=int, default=100)
    parser.add_argument('--target_vals', type=list, default=[-20000, -13500])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--target_val_for_policy', type=str, default=-13500)
    parser.add_argument('--load_data', type=str, default=True)
    parser.add_argument('--model_name', type=str, default='MDP')
    parser.add_argument('--load_model', type=str, default=False)

    args = parser.parse_args()
    parser.parse_args()
    experiment(variant=vars(args))
