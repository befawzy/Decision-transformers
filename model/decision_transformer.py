import torch.nn as nn
import torch
from transformers import DecisionTransformerModel


class DecisionTransformer(DecisionTransformerModel):

    """
    This model uses the huggingface implementation https://huggingface.co/docs/transformers/model_doc/decision_transformer#decision-transformer
    The model proposed in the original parper uses GPT architecture originally developed for language problems
    to model (Return_1, observation_1, action_1, Return_2, observation_2, ...)
    """

    def __init__(self, config):
        super().__init__(config)
        self.act_dim = config.act_dim
        self.state_dim = config.state_dim
        self.max_length = config.max_length
        self.predict_action = nn.Sequential(
            nn.Linear(config.hidden_size, config.act_dim))

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        action_preds = output[1]
        action_targets = kwargs["actions"]
        # make sure to set the attention_mask variable to True in the model configuration
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1,
                                            act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1,
                                                act_dim)[attention_mask.reshape(-1) > 0]
        # adding loss function for the sake of training. The nn.CrossEntropyLoss() class includes a softmax layer followed by cross entropy loss function
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(action_preds, action_targets)
        return {"loss": loss}

    def original_forward(self, *kwargs):
        state_preds, action_preds, return_preds = super().forward(*kwargs,
                                                                  return_dict=False)
        action_preds = torch.argmax(
            action_preds.reshape(-1, 3), dim=1).reshape(1, -1)
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # This implementation does not condition on past rewards
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            padding = self.max_length - states.shape[1]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [torch.zeros(padding), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat([torch.zeros((1, padding, self.state_dim)), states], dim=1).to(
                dtype=torch.float32)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1],
                             self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat([torch.zeros(
                (actions.shape[0], padding, self.act_dim)), actions], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([torch.zeros(
                (returns_to_go.shape[0], padding, 1)), returns_to_go], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([torch.zeros((timesteps.shape[0], padding),
                                  device=timesteps.device), timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.original_forward(
            states, actions, None, returns_to_go, timesteps, **kwargs)
        return action_preds[0, -1]
