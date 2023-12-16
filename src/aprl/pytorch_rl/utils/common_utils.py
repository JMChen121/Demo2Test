import torch

from aprl.pytorch_rl.utils.torch_util import to_device, tensor

from collections import namedtuple
import random


def set_abstracter(abstracter, states, actions, datatype, mapping_func, is_expert=False):
    with torch.no_grad():
        states_tensor = tensor(states).to(datatype)
        actions_tensor = tensor(actions).to(datatype)
        embeddings = mapping_func(states_tensor, actions_tensor, is_expert)
        grid_dict = abstracter.eval_diversity([embeddings.numpy()])
    return abstracter


def get_threshold(grid_dict, ratio=0.2):
    values = sorted(grid_dict.items(), key=lambda item: item[1], reverse=True)
    index = int(len(values) * ratio)
    return values[index][0], values[index][1]


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)


Transition_Wuji = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward', 'pol_id'))


class MemoryWuji(Memory):
    def __init__(self):
        super().__init__()

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition_Wuji(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition_Wuji(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition_Wuji(*zip(*random_batch))
