import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, target_state_dim, target_action_dim, source_state_dim, source_action_dim, hidden_size=None,
                 activation='tanh'):
        super().__init__()
        self.source_dim = source_state_dim + source_action_dim
        self.target_dim = target_state_dim + target_action_dim

        if hidden_size is None:
            self.hidden_size = [128]
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        # dim mapping (source/target to embedding)
        self.expert_embedding = nn.Linear(source_state_dim + source_action_dim, self.hidden_size[0]).double()
        self.generator_embedding = nn.Linear(target_state_dim + target_action_dim, self.hidden_size[0]).double()
        # hidden layers
        last_dim = self.hidden_size[0]
        for h_dim in self.hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, h_dim).double())
            last_dim = h_dim
        # discriminator classifier
        self.logic = nn.Linear(last_dim, 1).double()
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        prob = torch.sigmoid(self.logic(x))
        return prob

    def state_action_mapping(self, states, actions, is_expert=False):
        x = torch.cat([states, actions], 1)
        # expert embedding
        if x.shape[-1] == self.source_dim and is_expert:
            embedding = self.activation(self.expert_embedding(x))
        # generator embedding
        elif x.shape[-1] == self.target_dim and not is_expert:
            embedding = self.activation(self.generator_embedding(x))
        else:
            raise ValueError(f"The dimension of x should be same as the source or target, but get {x.shape[-1]}")
        return embedding
