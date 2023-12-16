# -*- coding: utf-8 -*-
import itertools
import warnings
from collections import defaultdict
from copy import deepcopy

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import geatpy as ea


def game_outcome(info):
    draw = True
    for i, agent_info in info.items():
        if not isinstance(i, int):
            continue
        if "winner" in agent_info:
            return i
    if draw:
        return -1


def _filter_dict(d, keys=None):
    if keys is None:
        return d
    else:
        keys = set(keys)
        present_keys = keys.intersection(d.keys())
        missing_keys = keys.difference(d.keys())
        res = {k: d[k] for k in present_keys}
        if len(missing_keys) != 0:
            warnings.warn("Missing expected keys: {}".format(missing_keys), stacklevel=2)
        return res


def record_timestep(wrapper, traj_dict, prev_obs, actions, values, rewards, dones, infos):
    env_data = {
        "observations": prev_obs,
        "actions": actions,
        "values": values,
        "rewards": rewards,
    }
    env_data = _filter_dict(env_data, wrapper.env_keys)

    # iterate over both agents over all environments in VecEnv
    iter_space = itertools.product(enumerate(traj_dict), range(wrapper.num_envs))

    for (dict_idx, agent_dicts), env_idx in iter_space:
        # in dict number dict_idx, record trajectories for agent number agent_idx
        agent_idx = wrapper.agent_indices[dict_idx]
        for key, val in env_data.items():
            # data_vals always have data for all agents (use agent_idx not dict_idx)
            if val is not None and val[agent_idx] is not None:
                agent_dicts[env_idx][key].append(val[agent_idx][env_idx])

        info_dict = infos[env_idx][agent_idx]
        info_dict = _filter_dict(info_dict, wrapper.info_keys)
        for key, val in info_dict.items():
            agent_dicts[env_idx][key].append(val)

        if dones[env_idx]:
            ep_ret = sum(agent_dicts[env_idx]["rewards"])
            wrapper.ea_win_trajectory[dict_idx]["episode_returns"].append(np.array([ep_ret]))
            wrapper.ea_win_trajectory[dict_idx]["episode_idx"].append(np.array([wrapper.episode_num]))
            wrapper.ea_win_trajectory[dict_idx]["prev_step_idx"].append(np.array([wrapper.step_times-1]))

            for key, val in agent_dicts[env_idx].items():
                # consolidate episode data and append to long-term data dict
                episode_key_data = np.array(val)
                wrapper.ea_win_trajectory[dict_idx][key].append(episode_key_data)
            agent_dicts[env_idx] = defaultdict(list)


class BestActionProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, venv_wrapper, action_a, action_b, params, origin_obs=None, origin_states=None):
        # print(f"Action lb: {round(action_a.min(), 2)}, ub: {round(action_a.max(), 2)}.")
        self.venv_wrapper = venv_wrapper
        self.origin_action = action_a  # value need to be optimized
        self.victim_action = action_b  # fixed value
        self.rollback_nums = params["rollback_frame_nums"]
        self.states = origin_states
        self.prev_obs = origin_obs

        # 在EA中胜利
        self.pop_dones = [False] * params["pop_num"]
        self.win_num = 0
        self.ea_win_traj = [[
            [defaultdict(list) for _ in range(venv_wrapper.num_envs)] for _ in venv_wrapper.agent_indices
        ]] * params["pop_num"]

        name = 'BestActionProblem'  # 初始化name（函数名称，可以随意设置）
        obj_dim = 1  # 初始化obj_dim（目标维数）
        is_min = [-1]  # is_min（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        var_dim = action_a.size  # 初始化var_dim（决策变量维数）
        var_types = [0] * var_dim  # 初始化var_types（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [params["lower_bound"]] * var_dim  # 决策变量下界
        ub = [params["upper_bound"]] * var_dim  # 决策变量上界
        lb_in = [1] * var_dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ub_in = [1] * var_dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 设置用多线程还是多进程, PoolType取值为'Process', 'Thread', 'None'
        self.PoolType = "None"
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(20)  # 设置池的大小
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            obj_dim,
                            is_min,
                            var_dim,
                            var_types,
                            lb,
                            ub,
                            lb_in,
                            ub_in,
                            evalVars=self.evalVars)

    def evalVars(self, action_a_pop):  # 目标函数，计算reward
        fs = np.zeros((len(action_a_pop), 1))

        policies = self.venv_wrapper.policies

        for pop_idx, action_a in enumerate(action_a_pop):
            if self.pop_dones[pop_idx]:
                # if any(self.pop_dones):
                break

            self.ea_win_traj[pop_idx] = [
                [defaultdict(list) for _ in range(self.venv_wrapper.num_envs)] for _ in self.venv_wrapper.agent_indices
            ]

            observations = deepcopy(self.prev_obs)
            states = deepcopy(self.states)
            dones = None
            # get origin state
            origin_state, origin_elapsed_steps = self.venv_wrapper.get_state()
            # run and rollback frames
            for step_time in range(self.rollback_nums):
                if self.pop_dones[pop_idx]:
                    break
                actions, new_states, new_values = [], [], []
                if step_time == 0:
                    # get actions from population of EA
                    actions = tuple([action_a.reshape(1, len(action_a)), self.victim_action])
                    new_states = states
                    new_values = None
                else:
                    # get actions from fixed policies
                    for policy_ind, (policy, obs, state) in enumerate(zip(policies, observations, states)):
                        try:
                            return_tuple = policy.predict_transparent(obs, state=state, mask=dones)
                            act, value, new_state, _, transparent_data = return_tuple
                        except AttributeError:
                            act, value, new_state, _ = policy.predict(obs, state=state, mask=dones)

                        actions.append(act)
                        new_states.append(new_state)
                        new_values.append(value)

                actions = tuple(actions)
                states = new_states

                # Step
                self.venv_wrapper.actions = actions
                self.venv_wrapper.venv.step_async(actions)
                observations, rewards, dones, infos = self.venv_wrapper.venv.step_wait()
                # Record timestep (frame) data
                record_timestep(self.venv_wrapper, self.ea_win_traj[pop_idx], self.prev_obs, actions, new_values,
                                rewards, dones, infos)

                # Calculate fitness from rewards.
                if step_time > 0:
                    fs[pop_idx] = fs[pop_idx] + rewards[0] - rewards[1]
                else:
                    positive_reward_names = ['reward_survive', 'reward_remaining', 'reward_forward', 'reward_center']
                    for k in positive_reward_names:
                        if k in infos[0][0]:
                            fs[pop_idx] = fs[pop_idx] + infos[0][0][k] - rewards[1]

                for _, (done, info) in enumerate(zip(dones, infos)):
                    if done:
                        self.pop_dones[pop_idx] = True
                        # find win
                        if game_outcome(info) == 0:
                            self.win_num += 1

                self.prev_obs = observations

            # set state to origin state
            self.venv_wrapper.set_state(origin_state, origin_elapsed_steps)

        return fs


def get_best_ant_action(venv_wrapper, init_actions, init_reward, dir_name, obs, origin_states, params, prophets=None):
    assert params is not None, 'Missing EA parameters!'
    # 实例化问题对象
    problem = BestActionProblem(venv_wrapper, init_actions[0], init_actions[1], params, obs, origin_states)
    # 构建算法
    prophet_vars = prophets if prophets is not None else np.repeat(init_actions[0], params["prophet_num"], axis=0)
    # algorithm = ea.soea_SEGA_templet(
    algorithm = ea.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='RI', Field=tuple([problem.varTypes, problem.ranges, problem.borders]),
                      NIND=params["pop_num"]),
        MAXGEN=100,  # 最大进化代数。
        logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
        trappedValue=0.2,  # 单目标优化陷入停滞的判断阈值（残差的阈值）。
        maxTrappedCount=5, )  # 进化停滞计数器最大上限值。
    # 求解
    res = ea.optimize(algorithm,
                      dirName=dir_name + "/ea/",
                      prophet=prophet_vars,
                      verbose=False,
                      drawing=0,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=True)
    res["ea_win_num"] = problem.win_num
    # print(res)
    return res
