import copy
import math
import os
import pickle
import time

from torch import nn

from aprl.abs.state_abstracter import StateAbstracter
from aprl.common.ActionPerturbation import action_perturb
from aprl.pytorch_rl.actor import Actor
from aprl.pytorch_rl.mlp_critic import Value
from aprl.pytorch_rl.mlp_discriminator import Discriminator
from aprl.pytorch_rl.mlp_policy import Policy
from aprl.pytorch_rl.mlp_policy_disc import DiscretePolicy
from aprl.pytorch_rl.ppo import ppo_step
from aprl.pytorch_rl.utils.common_utils import estimate_advantages, set_abstracter, get_threshold
from aprl.pytorch_rl.utils.torch_util import *


class GAIRL:

    def expert_reward(self, state, action, device=None):
        if device is None:
            device = self.device
        # state_action = tensor(np.hstack([state[0], action]), dtype=self.datatype).to(self.device)
        state, action = tensor(state, dtype=self.datatype).to(device), tensor([action], dtype=self.datatype).to(device)
        embedding = self.discriminator_net.state_action_mapping(state, action)
        with torch.no_grad():
            return self.expert_reward_weight * -math.log(self.discriminator_net(embedding)[0].item())

    def __init__(self, env, out_dir="", expert_dataset=None, seed=0, learning_rate=3e-4, log_std=-0.0, gpu_index=0,
                 num_threads=1, batchsize=2048, agent_id=0):
        self.batch_size = batchsize
        self.seed = seed
        self.out_dir = out_dir
        self.lr = learning_rate
        """seeding"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)
        """expert dataset"""
        self.expert_dataset = expert_dataset
        self.expert_dataset.init_dataloader(self.batch_size)
        self.source_state_dim = self.expert_dataset.observations.shape[1]
        self.source_action_dim = self.expert_dataset.actions.shape[1]
        self.expert_reward_weight = 0.05
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

        """define actor and critic"""
        if is_disc_action:
            self.policy_net = DiscretePolicy(self.state_dim, env.action_space.n)
        else:
            self.policy_net = Policy(self.state_dim, self.action_dim, self.device, log_std=log_std)
        self.value_net = Value(self.state_dim)
        self.discriminator_net = Discriminator(self.state_dim, self.action_dim,
                                               self.source_state_dim, self.source_action_dim)
        self.discriminator_fix = None
        self.discriminator_criterion = nn.BCELoss()
        to_device(self.device, self.policy_net, self.value_net, self.discriminator_net, self.discriminator_criterion)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator_net.parameters(), lr=self.lr)
        # optimization epoch number and batch size for PPO
        self.optim_epochs = 1
        # self.optim_batch_size = self.expert_dataset.dataloader.n_minibatches
        self.optim_batch_size = batchsize

        """create state abstracter"""
        self.key_state_abstracter = StateAbstracter(self.discriminator_net.hidden_size[0], 50, -1, 1,
                                                    self.source_action_dim, -2, 2)
        self.state_abstracter = StateAbstracter(self.discriminator_net.hidden_size[0], 50, -1, 1,
                                                self.source_action_dim, -2, 2)
        self.state_abstracter_win = StateAbstracter(self.discriminator_net.hidden_size[0], 50, -1, 1,
                                                    self.source_action_dim, -2, 2)

        """create agent for samples collection"""
        self.actor = Actor(env, self.device, self.datatype, self.policy_net, self.discriminator_fix, out_dir,
                           ks_abstracter=self.key_state_abstracter,
                           abstracter=self.state_abstracter, abstracter_win=self.state_abstracter_win,
                           custom_reward=self.expert_reward, perturb=action_perturb, num_threads=num_threads)
                           # custom_reward=self.expert_reward, perturb=None, num_threads=num_threads)

        """related with perturbation"""
        self.threshold = None
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

        """update discriminator"""
        st_expert, ac_expert = self.expert_dataset.get_next_batch(split=None)  # split=None, 'train', 'val'
        st_expert, ac_expert = torch.from_numpy(st_expert).to(self.datatype).to(self.device), \
                               torch.from_numpy(ac_expert).to(self.datatype).to(self.device)
        total_disc_loss = 0.0
        for _ in range(self.optim_epochs):
            e_embedding = self.discriminator_net.state_action_mapping(st_expert, ac_expert, is_expert=True)
            e_o = self.discriminator_net(e_embedding)
            g_embedding = self.discriminator_net.state_action_mapping(states, actions, is_expert=False)
            g_o = self.discriminator_net(g_embedding)

            self.optimizer_discriminator.zero_grad()
            disc_loss = self.discriminator_criterion(g_o, zeros((actions.shape[0], 1), device=self.device)) + \
                        self.discriminator_criterion(e_o, ones((ac_expert.shape[0], 1), device=self.device))
            disc_loss.backward()
            self.optimizer_discriminator.step()
            total_disc_loss += disc_loss

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

        return total_disc_loss, total_value_loss, total_policy_loss

    def learn(self, max_iter_num=100000, min_batch_size=1024, eval_batch_size=1024, log_interval=10,
              save_model_interval=100, render=False):
        assert self.expert_dataset is not None, "You must pass an expert dataset to GAIL for training"
        min_batch_size = self.batch_size
        eval_batch_size = self.batch_size / 2

        total_win_num, total_lose_num, total_tie_num, total_state_num, win_state_num = 0, 0, 0, 0, 0
        disc_loss, value_loss, policy_loss = float('inf'), float('inf'), float('inf')
        lr_dec_time = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

        for i_iter in range(max_iter_num):
            if i_iter <= lr_dec_time[-1] and i_iter in lr_dec_time:
                for params in self.optimizer_discriminator.param_groups:
                    params['lr'] -= self.lr * 0.1
                # self.expert_reward_weight -= 0.5 * 0.05
            """To use state embedding and abstract. Embedding is from state_action_mapping of discriminator_net"""
            if self.actor.discriminator is None and (disc_loss < 0.1 or i_iter >= 100):
            # if self.actor.discriminator is None:
                print(f"fix encoder of discriminator at iter: {i_iter}")
                self.discriminator_fix = copy.deepcopy(self.discriminator_net)
                self.discriminator_fix.requires_grad = False
                to_device('cpu', self.discriminator_fix)
                self.actor.discriminator = self.discriminator_fix
                # construct key state abstracter
                _ = set_abstracter(self.key_state_abstracter, self.expert_dataset.observations,
                                   self.expert_dataset.actions, self.datatype,
                                   self.actor.discriminator.state_action_mapping, is_expert=True)
                _, self.key_state_abstracter.threshold = get_threshold(self.key_state_abstracter.grid_dict, ratio=0.5)

            """generate multiple trajectories that reach the minimum batch_size (off-policy)"""
            # print(f"{i_iter}: start generate training samples")
            self.discriminator_net.to(torch.device('cpu'))
            batch, log = self.actor.collect_samples(self.agent_id, min_batch_size, render=render)
            self.discriminator_net.to(self.device)

            t0 = time.time()
            disc_loss, value_loss, policy_loss = self.update_params(batch, i_iter)
            t1 = time.time()

            """evaluate with deterministic action (remove noise for exploration)"""
            # print(f"{i_iter}: start collect evaluate samples")
            self.discriminator_net.to(torch.device('cpu'))
            _, log_eval = self.actor.collect_samples(self.agent_id, eval_batch_size, mean_action=True)
            self.discriminator_net.to(self.device)
            t2 = time.time()

            total_win_num += log['win_num'] + log_eval['win_num']
            total_lose_num += log['lose_num'] + log_eval['lose_num']
            total_tie_num += log['tie_num'] + log_eval['tie_num']
            total_state_num += log['num_steps'] + log_eval['num_steps']
            win_state_num += log['win_states_num'] + log_eval['win_states_num']

            if i_iter % log_interval == 0:
                total_num = total_win_num + total_lose_num + total_tie_num
                win_rate = round(total_win_num / total_num * 100, 2) if total_num != 0 else 0.0
                lose_rate = round(total_lose_num / total_num * 100, 2) if total_num != 0 else 0.0
                tie_rate = round(total_tie_num / total_num * 100, 2) if total_num != 0 else 0.0

                total_state_grids = len(self.state_abstracter.grid_dict.keys())
                total_win_grids = len(self.state_abstracter_win.grid_dict.keys())
                sate_div = round(total_state_grids / win_state_num * 100, 2) if win_state_num != 0 else 0.0
                win_div = round(total_win_grids / total_win_num * 100, 2) if total_win_num != 0 else 0.0
                pert_num = log['pert_num'] + log_eval['pert_num']
                pert_avg = round(pert_num / (log['num_episodes'] + log_eval['num_episodes']), 2)
                print(time.strftime('%Y%m%d-%H%M%S', time.localtime()) +
                      " {}\tT_sample {:.4f}, T_update {:.4f}, T_eval {:.4f}\t"
                      "train_disc_R {:.2f}, train_R {:.2f}; eval_disc_R {:.2f}, eval_R {:.2f}\t"
                      "d_loss {:.4f}, v_loss {:.4f}, p_loss {:.4f}".format(
                          i_iter, log['sample_time'], t1 - t0, t2 - t1,
                          log['avg_c_reward'], log['avg_reward'], log_eval['avg_c_reward'], log_eval['avg_reward'],
                          disc_loss, value_loss, policy_loss) +
                      f"\n\ttotal: {total_num}, win: {total_win_num} ({win_rate}%), "
                      f"lose:{total_lose_num} ({lose_rate}%), tie: {total_tie_num} ({tie_rate}%)\t"
                      f"win_states: {win_state_num}, states_grids: {total_state_grids} ({sate_div}%); "
                      f"win_nums: {total_win_num}, win_grids: {total_win_grids} ({win_div}%)\t"
                      f"pert_num: {log['pert_num'] + log_eval['pert_num']} ({pert_avg})")

            if save_model_interval > 0 and i_iter % save_model_interval == 0:
                # dire = os.path.join(self.out_dir, f"/checkpoint/{time.strftime('%Y%m%d-%H%M%S', time.localtime())}")
                # os.makedirs(dire, exist_ok=True)
                dire = self.out_dir
                to_device(torch.device('cpu'), self.policy_net, self.value_net, self.discriminator_net)
                time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime())
                pickle.dump((self.policy_net, self.value_net, self.discriminator_net, self.discriminator_fix),
                            open(os.path.join(dire, f"gail_{time_str}.pkl"), 'wb'))
                np.save(os.path.join(dire, f"state_grid_{time_str}.npy"), self.state_abstracter.grid_dict)
                np.save(os.path.join(dire, f"all_states_grid_{time_str}.npy"), self.state_abstracter.grid_states)
                np.save(os.path.join(dire, f"state_last_grid_{time_str}.npy"), self.state_abstracter_win.grid_dict)
                np.save(os.path.join(dire, f"all_states_last_grid_{time_str}.npy"), self.state_abstracter_win.grid_states)
                to_device(self.device, self.policy_net, self.value_net, self.discriminator_net)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

    def save(self, save_path):
        pickle.dump((self.policy_net, self.value_net, self.discriminator_net), open(save_path, 'wb'))
