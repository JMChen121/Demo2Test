import math
import os
import time

from aprl.envs.gym_compete import game_outcome
from aprl.pytorch_rl.utils.common_utils import Memory
from aprl.pytorch_rl.utils.torch_util import *

os.environ["OMP_NUM_THREADS"] = "1"


def sample_thread(actor, pid, queue, mean_action, render, min_batch_size, agent_id):
    env = actor.env
    policy = actor.policy
    discriminator = actor.discriminator
    datatype = actor.datatype
    ks_abstracter = actor.ks_abstracter
    abstracter = actor.abstracter
    abstracter_win = actor.abstracter_win
    custom_reward = actor.custom_reward
    running_state = actor.running_state
    perturb = actor.perturb
    out_dir = actor.out_dir

    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        if hasattr(env, 'np_random'):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
            env.env.np_random.seed(env.env.np_random.randint(5000) * pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0.0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0.0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    win_num, lose_num, tie_num = 0, 0, 0
    win_states_num = 0

    sample_states, sample_actions = None, None

    # observation, done
    observation = env.reset()
    done = False
    perturb_times = 0
    last_perturb = 0

    while num_steps < min_batch_size:
        # state = env.reset()
        # if running_state is not None:
        #     state = running_state(state)
        state = observation
        reward_episode = 0.0
        episode_steps = 0

        while not done:
            # predict action by policy
            state_var = tensor(state, dtype=datatype)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()

            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            sample_states = state_var if sample_states is None else torch.cat((sample_states, state_var))

            # Calculate reward from discriminator
            c_reward = 0
            if custom_reward is not None:
                c_reward = custom_reward(state, action, 'cpu')
                total_c_reward += c_reward
                min_c_reward = min(min_c_reward, c_reward)
                max_c_reward = max(max_c_reward, c_reward)

            # add perturbation to action
            is_perturbed = False
            # perturb at random state
            # random_perturb_state = np.random.randint(low=1, high=500, size=1)
            # random_perturb_state = np.random.randint(low=1, high=500, size=1)
            # random_perturb_state = np.random.randint(low=1, high=500, size=3)
            # random_perturb_state = np.random.randint(low=70, high=500, size=1)

            # sumoants
            # if 60 <= episode_steps <= 500 and episode_steps - last_perturb >= 15\
            # youshallnotpasshumans
            # if 60 <= episode_steps <= 140 and episode_steps - last_perturb >= 15\
            # runtogoalants
            # if 10 <= episode_steps <= 500 and episode_steps - last_perturb >= 15\
            # sumohumans
            if 70 <= episode_steps <= 150 and episode_steps - last_perturb >= 15\
                    and perturb is not None and discriminator is not None:
                action, is_perturbed, perturb_res = perturb(env, discriminator, out_dir, ks_abstracter, state_var,
                                                            tensor([action]), 'cpu',
                                                            must_perturb=False,
                                                            aux_fitness=actor.diversity_advance)
                if is_perturbed:
                    perturb_times = perturb_times + 1
                    last_perturb = episode_steps

            action_tensor = tensor([action])
            sample_actions = action_tensor if sample_actions is None else torch.cat((sample_actions, action_tensor))

            # step to next state in env
            next_state, reward, done, infos = env.step(action)
            episode_steps += 1
            reward_episode += reward[0]
            if running_state is not None:
                next_state = running_state(next_state)

            reward = reward + c_reward

            mask = 0 if done else 1
            # push sampled data to memory
            if len(memory) < min_batch_size and not is_perturbed:
                memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
            # game over
            if done or episode_steps > 500:
                # get winner
                winner = game_outcome(infos[0]) if episode_steps <= 500 else -1
                if winner == agent_id:
                    win_num += 1
                    win_states_num += sample_states.shape[0]
                    # eval fault diversity by abstracter
                    actor.win_states.append(sample_states)
                    actor.win_actions.append(sample_actions)
                    actor.win_last_states.append(state_var)
                    actor.win_last_actions.append(action_tensor)
                    if discriminator is not None:
                        with torch.no_grad():
                            if len(abstracter.grid_dict.keys()) == 0:
                                for i in range(len(actor.win_states)):
                                    embeddings_ep = discriminator.state_action_mapping(actor.win_states[i], actor.win_actions[i])
                                    grid_res = abstracter.eval_diversity([embeddings_ep.numpy()])
                                    embedding_last = discriminator.state_action_mapping(actor.win_last_states[i], actor.win_last_actions[i])
                                    grid_res_win = abstracter_win.eval_diversity([embedding_last.numpy()])
                            else:
                                embeddings_ep = discriminator.state_action_mapping(sample_states, sample_actions)
                                grid_res = abstracter.eval_diversity([embeddings_ep.numpy()])
                                embedding_last = discriminator.state_action_mapping(state_var, action_tensor)
                                grid_res_win = abstracter_win.eval_diversity([embedding_last.numpy()])

                elif winner == 1 - agent_id:
                    lose_num += 1
                elif winner == -1:
                    tie_num += 1

                done = False
                sample_states, sample_actions = None, None
                break

            state = next_state

        # log stats
        num_steps += episode_steps
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['win_states_num'] = win_states_num
    log['total_reward'] = total_reward
    log['avg_reward'] = (total_reward / num_episodes)
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    log['win_num'] = win_num
    log['lose_num'] = lose_num
    log['tie_num'] = tie_num
    log['pert_num'] = perturb_times
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


class Actor:

    def __init__(self, env, device, datatype, policy, discriminator=None, out_dir=None, ks_abstracter=None,
                 abstracter=None, abstracter_win=None, custom_reward=None, perturb=None, running_state=None,
                 num_threads=1):
        self.env = env
        self.device = device
        self.policy = policy
        self.discriminator = discriminator
        self.out_dir = out_dir
        self.custom_reward = custom_reward
        self.perturb = perturb
        self.running_state = running_state
        self.num_threads = num_threads
        self.datatype = datatype
        self.ks_abstracter = ks_abstracter
        self.abstracter = abstracter
        self.abstracter_win = abstracter_win

        self.win_states, self.win_actions = [], []
        self.win_last_states, self.win_last_actions = [], []

    def collect_samples(self, agent_id, min_batch_size, mean_action=False, render=False):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))

        memory, log = sample_thread(self, 0, None, mean_action, render, thread_batch_size, agent_id)
        batch = memory.sample()

        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        # log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        # log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        # log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log

    def diversity_advance(self, state, action):
        action_tensor = action if torch.is_tensor(action) else tensor([action])
        state_tensor = state if torch.is_tensor(state) else tensor(state)
        with torch.no_grad():
            embedding = self.discriminator.state_action_mapping(state_tensor, action_tensor)
            grid_id, _ = self.abstracter.get_state_grid_ids(embedding)
            grid_id = grid_id[0]
            if grid_id in self.abstracter.grid_dict:
                return 0
            else:
                return 1
