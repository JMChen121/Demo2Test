# -*- coding: utf-8 -*-
from copy import deepcopy
from multiprocessing.dummy import Pool as ThreadPool

import geatpy as ea
import numpy as np
import torch

from aprl.common.mujoco import MujocoState


def game_outcome(info):
    draw = True
    for i, agent_info in info.items():
        if not isinstance(i, int):
            continue
        if "winner" in agent_info:
            return i
    if draw:
        return -1


def judge_state(embedding, key_state_abstracter, threshold=0):
    grid_id, _ = key_state_abstracter.get_state_grid_ids(embedding)
    grid_id = grid_id[0]
    if grid_id in key_state_abstracter.grid_dict:
        return key_state_abstracter.grid_dict[grid_id], key_state_abstracter.grid_dict[grid_id] >= threshold
    else:
        return 0, 0 >= threshold


def get_env_state(env):
    return MujocoState.from_mjdata(env.env_scene.data).flatten(), \
           getattr(env, "_elapsed_steps") if hasattr(env, "_elapsed_steps") else None


def set_env_state(env, x, _elapsed_steps=None):
    """Restores q_pos and q_vel, calling forward() to derive other values."""
    state = MujocoState.from_flattened(x, env.env_scene)
    state.set_mjdata(env.env_scene.data)
    if _elapsed_steps is not None:
        setattr(env, "_elapsed_steps", _elapsed_steps)
    env.env_scene.model.forward()  # put mjData in consistent state


class BestPerturbationProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, venv_wrapper, state_tensor, action_a, params, origin_obs=None, aux_fitness=None):
        # print(f"Action lb: {round(action_a.min(), 2)}, ub: {round(action_a.max(), 2)}.")
        self.venv_wrapper = venv_wrapper
        self.origin_action = action_a  # value need to be optimized
        self.rollback_nums = params["rollback_frame_nums"]
        self.state_tensor = state_tensor
        self.prev_obs = origin_obs
        self.aux_fitness = aux_fitness

        # 在EA中胜利
        self.pop_dones = [False] * params["pop_num"]
        self.win_num = 0

        name = 'BestActionProblem'  # 初始化name（函数名称，可以随意设置）
        obj_dim = 1  # 初始化obj_dim（目标维数）
        is_min = [-1]  # is_min（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        var_dim = action_a.shape[1]  # 初始化var_dim（决策变量维数）
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

        for pop_idx, action_a in enumerate(action_a_pop):
            if self.pop_dones[pop_idx]:
                break

            observations = deepcopy(self.prev_obs)
            states = deepcopy(self.state_tensor)
            # get origin state
            origin_state, origin_elapsed_steps = get_env_state(self.venv_wrapper.unwrapped.envs[0])
            # run and rollback frames
            for step_time in range(self.rollback_nums):
                if self.pop_dones[pop_idx]:
                    break
                new_states = []
                if step_time == 0:
                    new_states = states
                else:
                    # get actions from policies
                    pass

                states = new_states

                # Step
                next_state, rewards, dones, infos = self.venv_wrapper.step(action_a)

                # Calculate fitness from rewards.
                if step_time > 0:
                    fs[pop_idx] = fs[pop_idx] + rewards[0] * 0.5
                else:
                    positive_reward_names = ['reward_survive', 'reward_remaining', 'reward_forward', 'reward_center']
                    for k in positive_reward_names:
                        if k in infos[0][0]:
                            fs[pop_idx] = fs[pop_idx] + (infos[0][0][k] - infos[0][1][k]) * 0.2

                # Calculate fitness from diversity of abstraction grid.
                if self.aux_fitness is not None:
                    fs[pop_idx] = fs[pop_idx] + self.aux_fitness(states, action_a)

                for _, (done, info) in enumerate(zip(dones, infos)):
                    if done:
                        self.pop_dones[pop_idx] = True
                        # find win
                        if game_outcome(info) == 0:
                            self.win_num += 1

                self.prev_obs = observations

            # set state to origin state
            set_env_state(self.venv_wrapper.unwrapped.envs[0], origin_state, origin_elapsed_steps)

        return fs


def get_perturbed_action(venv, state_tensor, init_action, dir_name, params, prophets=None, aux_fitness=None):
    assert params is not None, 'Missing EA parameters!'
    # 实例化问题对象
    problem = BestPerturbationProblem(venv, state_tensor, init_action, params, aux_fitness=aux_fitness)
    # 构建算法
    prophet_vars = prophets if prophets is not None else np.repeat(init_action, params["prophet_num"], axis=0)
    # algorithm = ea.soea_SEGA_templet(
    algorithm = ea.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='RI', Field=tuple([problem.varTypes, problem.ranges, problem.borders]),
                      NIND=params["pop_num"]),
        MAXGEN=1,  # 最大进化代数。
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


def action_perturb(env, discriminator, out_dir, key_state_abstracter, state_tensor, action_tensor,
                   device=None, must_perturb=False, aux_fitness=None):
    action = action_tensor.numpy()
    # if must_perturb:
    #     do_ea = must_perturb
    # embedding
    with torch.no_grad():
        embedding = discriminator.state_action_mapping(state_tensor, action_tensor)
    occur_times, do_ea = judge_state(embedding, key_state_abstracter, key_state_abstracter.threshold)

    # do_ea = True    # TODO: delete this line
    if do_ea:
        # # ea param
        # ea_params = {
        #     "pop_num": 25,
        #     "prophet_num": 1,
        #     "rollback_frame_nums": 1,
        #     "lower_bound": -2,
        #     "upper_bound": 2
        # }
        # action_res = get_perturbed_action(env, state_tensor, action, out_dir, ea_params, aux_fitness=aux_fitness)
        # return action_res['Vars'][0], do_ea, action_res

        # random perturbed
        scale = 0
        if action.size == 8:
            scale = 0.2
        elif action.size == 17:
            scale = 0.4
        assert scale != 0, "Shape of action error!"
        noise = np.random.normal(0, scale, action.shape)
        noise_action = action + noise
        return noise_action[0], do_ea, None
    else:
        return action[0], do_ea, None
