import numpy as np
import torch
# import wandb
import argparse
import pickle
import time
import torch
from torch import nn
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from decision_transformer import DecisionTransformer
from transformers import DecisionTransformerModel
from data_collator import DecisionTransformerDataCollator
from test_data_generator import generate_dataset
from evaluate_rtg import load_model,evaluate
# wandb.login()


# def main(variant):
#load env parameters for data generation
    # env_path= 'env/mean_env_params.pickle'
# env_path= variant['env_path']
# trajs=variant['num_of_trajectories']
# epsilon=variant['epsilon']
env_path='env/mean_env_params.pickle'
model_name='MDP'
n_trajs=1000
epsilon=0
epochs=20
data=generate_dataset(n_trajs,'env/mean_env_params.pickle',epsilon)
print('done with data generation')
print(data)
dataset_path = f'data/dataset-{n_trajs}-{epsilon}.pkl'
with open(dataset_path, 'wb') as f:
			pickle.dump(data, f)

collator = DecisionTransformerDataCollator(data,model_name=model_name)
# np.array(nn.functional.one_hot(
#                 torch.as_tensor(data['actions'])[0], 3))
config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim,
                                       max_length=collator.max_len,
                                       max_ep_len= collator.max_ep_len,
                                       hidden_size=128,
                                       n_layer=3,
                                       n_head=1,
                                       n_inner=4*128,
                                       n_positions=1024,
                                       resid_pdrop=0.1,
                                       attn_pdrop=0.1
                                       )                                       
model = DecisionTransformer(config)
model = model.to(device='cpu')
output_dir=f'output/{model_name}/samples_{n_trajs}/epsilon_{epsilon}/epochs_{epochs}'
training_args = TrainingArguments(
    output_dir=output_dir,
    remove_unused_columns=False,
    num_train_epochs=epochs,
    per_device_train_batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-4,
    lr_scheduler_type='linear',
    optim="adamw_torch",
    max_grad_norm=0.25,
)
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        data_collator=collator,
    )
# with wandb.init(project='decision-transformer',config=training_args):
#     config= wandb.config


# log_to_wandb=True
# wandb_config = dict(
#     epochs=epochs,
#     epsilon=epsilon,
#     num_of_trajectories=n_trajs,
#     model_name=model_name,
#     batch_size=256,
#     learning_rate=1e-3,
#     context_length=collator.max_len,
#     dataset="fractal-values",
#     architecture="decision-transformer")
# if log_to_wandb:
#         wandb.init(
#             dir=output_dir,
#             project='decision-transformer',
#             config=wandb_config
#         )
# wandb.watch(model)        

trainer.train()
# print('saving to: ')
# print(output_dir)

torch.save(model.state_dict(),output_dir+'/pt')
config.save_pretrained(output_dir)

DT=load_model(output_dir)
untrained_model=DecisionTransformer(config)
# torch.argmax(actions.reshape(-1,3),dim=1,keepdim=True)
DT.eval()
targets= np.arange(-30000,-5000,1000)
actions=torch.zeros((targets.shape[0],50))
states=torch.zeros((targets.shape[0],50))
actual_return=torch.zeros(targets.shape[0])
for i, target in enumerate(targets):
    actions[i], states[i], actual_return[i]=evaluate(DT,target, env_path=env_path)
# actions[0], states[0], actual_return[0]=evaluate(model,targets[-1], env_path=env_path)
for i in range(actions.shape[0]):
    plt.figure()
    plt.plot(np.arange(50),actions[i], label= 'actions', color='r', alpha=0.5,marker = '.', markersize=10)
    plt.plot(np.arange(50),states[i], label= 'states', color='b', alpha=0.5,marker = '.', markersize=10)
    plt.legend()


for i in range(actions.shape[0]):
    plt.figure()
    plt.plot(np.arange(50),np.argmax(data['actions'][i],axis=1), label= 'actions', color='r', alpha=0.5,marker = '.', markersize=10 )
    plt.plot(np.arange(50),np.argmax(data['states'][i],axis=1), label= 'states', color='b', alpha=0.5,marker = '.', markersize=10 )


plt.plot(np.arange(data.shape[0])[:20],np.sum(data['rewards'],axis=1)[:20], label= 'actions', color='r', alpha=0.5,marker = '.', markersize=10 )
    

# output=trainer.compute_loss(model=model,inputs=data)
# if log_to_wandb:
#     wandb.log(output)
with open(dataset_path, 'rb') as f:
        trajs = pickle.load(f)
    #evaluation
# num_eval_episodes=arguments['num_eval_episodes']
# target_rewards=arguments['target_reward']    
    # group_name = f'{exp_prefix}-{env_name}-{dataset}'
    # exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    # log_to_wandb=True
    # if log_to_wandb:
    #     wandb.init(
    #         group=group_name,
    #         project='decision-transformer',
    #         config=variant
    #     )
    #     # wandb.watch(model)  # wandb has some bug
    # for iter in range(variant['max_iters']):
    #     outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
    #     if log_to_wandb:
    #         wandb.log(outputs)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env_path', type=str, default='env/mean_env_params.pickle')
#     # medium, medium-replay, medium-expert, expert
#     parser.add_argument('--num_of_trajectories', type=int, default=10000)
#     parser.add_argument('--K', type=int, default=20)
#     parser.add_argument('--pct_traj', type=float, default=1.)
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--embed_dim', type=int, default=128)
#     parser.add_argument('--n_layer', type=int, default=12)
#     parser.add_argument('--epsilon', type=float, default=0.3)
#     parser.add_argument('--n_head', type=int, default=1)
#     parser.add_argument('--max_ep_len', type=int, default=50)
#     parser.add_argument('--activation_function', type=str, default='softmax')
#     parser.add_argument('--dropout', type=float, default=0.1)
#     parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
#     parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
#     parser.add_argument('--warmup_steps', type=int, default=10000)
#     parser.add_argument('--num_eval_episodes', type=int, default=100)
#     parser.add_argument('--max_iters', type=int, default=10)
#     parser.add_argument('--target_reward', type=list, default=[10,20])
#     parser.add_argument('--num_steps_per_iter', type=int, default=10000)
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
#     args = parser.parse_args()
#     parser.parse_args()
#     main(variant=vars(args))