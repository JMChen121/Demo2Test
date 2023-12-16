import math
import os
import pickle
import time

import numpy as np
import torch

from aprl.abs.state_abstracter import StateAbstracter
from aprl.pytorch_rl.actor import Actor
from aprl.pytorch_rl.mlp_critic import Value
from aprl.pytorch_rl.mlp_discriminator import Discriminator
from aprl.pytorch_rl.mlp_policy import Policy
from aprl.pytorch_rl.mlp_policy_disc import DiscretePolicy
from aprl.pytorch_rl.utils.common_utils import estimate_advantages
from aprl.pytorch_rl.utils.torch_util import to_device, LongTensor


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, clip_epsilon=0.2, l2_reg=1e-3):
    """"""
    """update value net"""
    total_value_loss = 0.0
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()
        total_value_loss += value_loss

    """update policy net"""
    log_probs = policy_net.get_log_prob(states, actions)
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

    return total_value_loss, policy_surr


class PPO:
    def __init__(self, env, out_dir="", expert_dataset=None, seed=0, learning_rate=5e-3, log_std=-0.0, gpu_index=0,
                 num_threads=1, batchsize=2048, model_path=None, agent_id=0):
        self.batch_size = batchsize
        self.seed = seed
        self.out_dir = out_dir
        # model_path = "data/baselines/20231203_212933-gairl_SumoAnts-v0/gail_20231203-220616.pkl"
        # model_path = "data/baselines/20231205_205705-gairl_YouShallNotPassHumans-v0/gail_20231205-225041.pkl"
        model_path = "data/baselines/20231207_095824-gairl_RunToGoalAnts-v0/gail_20231207-101855.pkl"
        # model_path = "data/baselines/20231204_233105-gairl_SumoHumans-v0/gail_20231204-235230.pkl"

        """expert dataset"""
        self.expert_dataset = expert_dataset
        self.expert_dataset.init_dataloader(self.batch_size)
        self.source_state_dim = self.expert_dataset.observations.shape[1]
        self.source_action_dim = self.expert_dataset.actions.shape[1]

        """environment"""
        self.agent_id = agent_id
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        # self.running_state = ZFilter((self.state_dim,), clip=5)
        self.running_state = None
        is_disc_action = len(env.action_space.shape) == 0
        self.action_dim = 1 if is_disc_action else env.action_space.shape[0]

        """torch setting"""
        self.datatype = torch.float64
        torch.set_default_dtype(self.datatype)
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_index)
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')

        """seeding"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

        """define actor and critic"""
        if is_disc_action:
            self.policy_net = DiscretePolicy(self.state_dim, env.action_space.n)
        else:
            self.policy_net = Policy(self.state_dim, env.action_space.shape[0], self.device, log_std=log_std)
        self.value_net = Value(self.state_dim)
        if model_path is None:
            self.discriminator_fix = Discriminator(self.state_dim, self.action_dim,
                                                   self.state_dim, self.action_dim)
            self.discriminator_fix.requires_grad = False
            to_device('cpu', self.discriminator_fix)
        else:
            print(f"read discriminator from: {model_path}")
            _, _, _, self.discriminator_fix = pickle.load(open(model_path, "rb"))
            to_device('cpu', self.discriminator_fix)
        to_device(self.device, self.policy_net, self.value_net)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)

        # optimization epoch number and batch size for PPO
        self.optim_epochs = 1
        # self.optim_batch_size = self.expert_dataset.dataloader.n_minibatches
        self.optim_batch_size = batchsize
        """create state abstracter"""

        self.state_abstracter = StateAbstracter(self.discriminator_fix.hidden_size[0], 50, -1, 1,
                                                self.action_dim, -2, 2)
        self.state_abstracter_win = StateAbstracter(self.discriminator_fix.hidden_size[0], 50, -1, 1,
                                                    self.action_dim, -2, 2)

        """create agent for samples collection"""
        # self.actor = Actor(env, self.policy_net, self.device, running_state=self.running_state, num_threads=num_threads,
        #                    datatype=self.datatype)
        self.actor = Actor(env, self.device, self.datatype, self.policy_net, None,
                           abstracter=self.state_abstracter, abstracter_win=self.state_abstracter_win,
                           num_threads=num_threads)
        """"""

    def update_params(self, batch, i_iter, gamma=0.99, tau=0.95):
        states = torch.squeeze(torch.from_numpy(np.stack(batch.state)).to(self.datatype).to(self.device), dim=1)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.datatype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.datatype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.datatype).to(self.device)
        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, gamma, tau, self.device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        total_value_loss, total_policy_loss = 0.0, 0.0
        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                value_loss, policy_loss = ppo_step(self.policy_net, self.value_net, self.optimizer_policy,
                                                   self.optimizer_value, 1, states_b, actions_b, returns_b,
                                                   advantages_b, fixed_log_probs_b)
                total_value_loss += value_loss
                total_policy_loss += policy_loss

        return total_value_loss, total_policy_loss

    def learn(self, max_iter_num=100000, min_batch_size=1024, eval_batch_size=1024, log_interval=10,
              save_model_interval=100, render=False):
        min_batch_size = self.batch_size
        eval_batch_size = self.batch_size / 2

        total_win_num, total_lose_num, total_tie_num, total_state_num,  win_state_num = 0, 0, 0, 0, 0

        for i_iter in range(max_iter_num):
            """To use state embedding and abstract. Embedding is from state_action_mapping of discriminator_net"""
            if i_iter >= 0 and self.actor.discriminator is None:
                self.actor.discriminator = self.discriminator_fix
            """generate multiple trajectories that reach the minimum batch_size"""
            batch, log = self.actor.collect_samples(self.agent_id, min_batch_size, render=render)

            t0 = time.time()
            value_loss, policy_loss = self.update_params(batch, i_iter)
            t1 = time.time()

            """evaluate with determinstic action (remove noise for exploration)"""
            _, log_eval = self.actor.collect_samples(self.agent_id, eval_batch_size, mean_action=True)
            t2 = time.time()

            total_win_num += log['win_num'] + log_eval['win_num']
            total_lose_num += log['lose_num'] + log_eval['lose_num']
            total_tie_num += log['tie_num'] + log_eval['tie_num']
            total_state_num += log['num_steps'] + log_eval['num_steps']
            win_state_num += log['win_states_num'] + log_eval['win_states_num']

            if i_iter % log_interval == 0:
                total_num = total_win_num + total_lose_num + total_tie_num
                win_rate = round(total_win_num / total_num * 100, 2) if total_num != 0 else 0.0
                loss_rate = round(total_lose_num / total_num * 100, 2) if total_num != 0 else 0.0
                tie_rate = round(total_tie_num / total_num * 100, 2) if total_num != 0 else 0.0

                total_state_grids = len(self.state_abstracter.grid_dict.keys())
                total_win_grids = len(self.state_abstracter_win.grid_dict.keys())
                sate_div = round(total_state_grids / win_state_num * 100, 2) if win_state_num != 0 else 0.0
                win_div = round(total_win_grids / total_win_num * 100, 2) if total_win_num != 0 else 0.0
                print(time.strftime('%Y%m%d-%H%M%S', time.localtime()) +
                      " {}\tT_sample {:.4f}, T_update {:.4f}, T_eval {:.4f}\t"
                      "train_R_min {:.2f}, train_R_max {:.2f}, train_R {:.2f}; eval_R {:.2f}\t"
                      "v_loss {:.4f}, policy_loss {:.4f}\t".
                      format(i_iter, log['sample_time'], t1 - t0, t2 - t1, log['min_reward'], log['max_reward'],
                             log['avg_reward'], log_eval['avg_reward'], value_loss, policy_loss) +
                      f"\n\ttotal: {total_num}, win: {total_win_num} ({win_rate}%), "
                      f"loss:{total_lose_num} ({loss_rate}%), tie: {total_tie_num} ({tie_rate}%)\t"
                      f"win_states: {win_state_num}, states_grids: {total_state_grids} ({sate_div}%); "
                      f"win_nums: {total_win_num}, win_grids: {total_win_grids} ({win_div}%)")

            if save_model_interval > 0 and (i_iter + 1) % save_model_interval == 0:
                dire = self.out_dir
                to_device(torch.device('cpu'), self.policy_net, self.value_net)
                time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime())
                pickle.dump((self.policy_net, self.value_net),
                            open(os.path.join(dire, f"ppo_{time_str}.pkl"), 'wb'))
                np.save(os.path.join(dire, f"state_grid_{time_str}.npy"), self.state_abstracter.grid_dict)
                np.save(os.path.join(dire, f"all_states_grid_{time_str}.npy"), self.state_abstracter.grid_states)
                np.save(os.path.join(dire, f"state_last_grid_{time_str}.npy"), self.state_abstracter_win.grid_dict)
                np.save(os.path.join(dire, f"all_states_last_grid_{time_str}.npy"), self.state_abstracter_win.grid_states)
                to_device(self.device, self.policy_net, self.value_net)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

    def save(self, save_path):
        pickle.dump((self.policy_net, self.value_net), open(save_path, 'wb'))
