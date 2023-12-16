import random
from collections import defaultdict
import itertools
import os
from os import path as osp
import warnings

import gym
from gym import Wrapper
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from scipy.spatial import distance

from aprl.common.key_frame_tools import find_closed_observation
from aprl.common.multi_monitor import MultiMonitor
from aprl.envs.multi_agent import SingleToMulti, VecMultiWrapper, VecMultiResettableWrapper
from aprl.common.ActionProblem import get_best_ant_action


def random_unit(p):
    assert 0 <= p <= 1, "概率P的值应该处在[0,1]之间！"
    if p == 0:  # 概率为0，直接返回False
        return False
    if p == 1:  # 概率为1，直接返回True
        return True
    p_digits = len(str(p).split(".")[1])
    interval_begin = 1
    interval__end = pow(10, p_digits)
    R = random.randint(interval_begin, interval__end)
    if float(R) / interval__end < p:
        return True
    else:
        return False


class VideoWrapper(Wrapper):
    """Creates videos from wrapped environment by called render after each timestep."""

    def __init__(self, env, directory, single_video=True):
        """

        :param env: (gym.Env) the wrapped environment.
        :param directory: the output directory.
        :param single_video: (bool) if True, generates a single video file, with episodes
                             concatenated. If False, a new video file is created for each episode.
                             Usually a single video file is what is desired. However, if one is
                             searching for an interesting episode (perhaps by looking at the
                             metadata), saving to different files can be useful.
        """
        super(VideoWrapper, self).__init__(env)
        self.episode_id = 0
        self.video_recorder = None
        self.single_video = single_video

        self.directory = osp.abspath(directory)

        # Make sure to not put multiple different runs in the same directory,
        # if the directory already exists
        error_msg = (
            "You're trying to use the same directory twice, "
            "this would result in files being overwritten"
        )
        assert not os.path.exists(self.directory), error_msg
        os.makedirs(self.directory, exist_ok=True)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            winners = [i for i, d in info.items() if "winner" in d]
            metadata = {"winners": winners}
            self.video_recorder.metadata.update(metadata)
        self.video_recorder.capture_frame()
        return obs, rew, done, info

    def reset(self):
        self._reset_video_recorder()
        self.episode_id += 1
        return self.env.reset()

    def _reset_video_recorder(self):
        """Called at the start of each episode (by _reset). Always creates a video recorder
        if one does not already exist. When a video recorder is already present, it will only
        create a new one if `self.single_video == False`."""
        if self.video_recorder is not None:
            # Video recorder already started.
            if not self.single_video:
                # We want a new video for each episode, so destroy current recorder.
                self.video_recorder.close()
                self.video_recorder = None

        if self.video_recorder is None:
            # No video recorder -- start a new one.
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path=osp.join(self.directory, "video.{:06}".format(self.episode_id)),
                metadata={"episode_id": self.episode_id},
            )

    def close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super(VideoWrapper, self).close()


def _filter_dict(d, keys):
    """Filter a dictionary to contain only the specified keys.

    If keys is None, it returns the dictionary verbatim.
    If a key in keys is not present in the dictionary, it gives a warning, but does not fail.

    :param d: (dict)
    :param keys: (iterable) the desired set of keys; if None, performs no filtering.
    :return (dict) a filtered dictionary."""
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


def get_prophets(prophet_nums, policy, ob, state, mask=None):
    actions = np.empty((prophet_nums, policy.action_space.shape[0]))
    for i in range(prophet_nums):
        try:
            return_tuple = policy.predict_transparent(ob, state=state, mask=mask)
            act, _, new_state, _, transparent_data = return_tuple
        except AttributeError:
            act, _, new_state, _ = policy.predict(ob, state=state, mask=mask)
        actions[i] = act
    return actions


class TrajectoryResettableRecorder(VecMultiResettableWrapper):

    def __init__(self, venv, agent_indices=None, save_dir=None, env_keys=None, info_keys=None):
        super().__init__(venv)

        if agent_indices is None:
            self.agent_indices = range(self.num_agents)
        elif isinstance(agent_indices, int):
            self.agent_indices = [agent_indices]
        self.save_dir = save_dir
        self.env_keys = env_keys
        self.info_keys = info_keys

        self.traj_dicts = [
            [defaultdict(list) for _ in range(self.num_envs)] for _ in self.agent_indices
        ]
        self.full_traj_dicts = [defaultdict(list) for _ in self.agent_indices]
        self.prev_obs = None
        self.actions = None
        self.values_record = None
        self.episode_num = 0
        # resettable
        self.step_times = 0
        self.policies = None
        self.temp_states = None
        self.temp_observations = None
        # ea
        self.need_ea = False
        self.ea_params = None
        self.last_ea_f = 0
        self.ea_times_total = 0
        self.ea_win_num = 0
        self.ea_win_trajectory = [defaultdict(list) for _ in self.agent_indices]

        self.key_frames_info = None
        self.guide = False

    def step_async(self, actions):
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        # if self.need_ea and self.ea_params["start_frame"] <= self.step_times <= self.ea_params["end_frame"] and\
        #         (self.step_times - self.ea_params["start_frame"]) % self.ea_params["interval"] == 0:
        do_ea = False
        if self.need_ea and self.ea_params["start_frame"] <= self.step_times <= self.ea_params["end_frame"] and \
           self.step_times - self.last_ea_f >= self.ea_params["interval"]:
            do_ea = True
            # Judge this frame is key frmae or not, according to distances of observation and candidates from Highlight
            if self.guide:
                _, do_ea = find_closed_observation(self.temp_observations[0][0], self.key_frames_info["observations"],
                                                   distance.euclidean, threshold=self.ea_params["key_threshold"])
            if do_ea:
                curr_state, curr_elapsed_steps = self.get_state()
                o_observations, o_reward, o_dones, o_infos = self.venv.step_wait()
                victim_act = self.actions[1]
                self.set_state(curr_state, curr_elapsed_steps)

                # get prophet
                prophet_acts = get_prophets(self.ea_params["prophet_num"] - 1, self.policies[0],
                                            self.temp_observations[0],
                                            self.temp_states[0])
                prophet_acts = np.concatenate((self.actions[0], prophet_acts))
                # get better action by EA
                action_res = get_best_ant_action(self, self.actions, o_reward, self.save_dir, self.prev_obs,
                                                 self.temp_states, self.ea_params, prophet_acts)
                if action_res["ea_win_num"] != 0:
                    self.ea_win_num += action_res["ea_win_num"]
                # reset actions
                better_action_a = action_res['Vars']
                new_actions = tuple([better_action_a, victim_act])
                self.step_async(new_actions)
                observations, rewards, dones, infos = self.venv.step_wait()
                self.last_ea_f = self.step_times
                self.ea_times_total += 1
                print(f"Old rewards: {o_infos[0]}")
                print(f"New rewards: {infos[0]}")
                print(f"---Action EA at frame: {self.last_ea_f}.\t"
                      f"Final fitness: {action_res['ObjV']}.\t"
                      f"Act lb: {round(float(self.actions[0].min()),2)}, ub: {round(float(self.actions[0].max()),2)}.\t"
                      f"Reward change: ({round(float(o_reward[0][0]), 6)}, {round(float(o_reward[1][0]), 6)}) --> "
                      f"({round(float(rewards[0][0]), 6)}, {round(float(rewards[1][0]), 6)})---")

        # actually step_wait
        if not do_ea:
            observations, rewards, dones, infos = self.venv.step_wait()

        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                print(f"Done at {self.step_times}. Info: {info[0]}; {info[1]}")
                observations = self.reset()
                self.last_ea_f = self.step_times = 0
                self.episode_num += 1
            else:
                self.step_times += 1
        # record timestep (frame) data
        self.record_timestep_data(self.prev_obs, self.actions, self.values_record, rewards, dones, infos)
        self.prev_obs = observations
        return observations, rewards, dones, infos

    def reset(self):
        observations = self.venv.reset()
        self.prev_obs = observations
        return observations

    def record_extra_data(self, data, agent_idx):

        if agent_idx not in self.agent_indices:
            return
        else:
            dict_index = self.agent_indices.index(agent_idx)

        for env_idx in range(self.num_envs):
            for key in data.keys():
                self.traj_dicts[dict_index][env_idx][key].append(np.squeeze(data[key]))

    def record_timestep_data(self, prev_obs, actions, values, rewards, dones, infos):

        env_data = {
            "observations": prev_obs,
            "actions": actions,
            "values": values,
            "rewards": rewards,
        }
        env_data = _filter_dict(env_data, self.env_keys)

        # iterate over both agents over all environments in VecEnv
        iter_space = itertools.product(enumerate(self.traj_dicts), range(self.num_envs))
        for (dict_idx, agent_dicts), env_idx in iter_space:
            # in dict number dict_idx, record trajectories for agent number agent_idx
            agent_idx = self.agent_indices[dict_idx]
            for key, val in env_data.items():
                # data_vals always have data for all agents (use agent_idx not dict_idx)
                if val is not None and val[agent_idx] is not None:
                    agent_dicts[env_idx][key].append(val[agent_idx][env_idx])

            info_dict = infos[env_idx][agent_idx]
            info_dict = _filter_dict(info_dict, self.info_keys)
            for key, val in info_dict.items():
                agent_dicts[env_idx][key].append(val)

            if dones[env_idx]:
                ep_ret = sum(agent_dicts[env_idx]["rewards"])
                self.full_traj_dicts[dict_idx]["episode_returns"].append(np.array([ep_ret]))

                for key, val in agent_dicts[env_idx].items():
                    # consolidate episode data and append to long-term data dict
                    episode_key_data = np.array(val)
                    self.full_traj_dicts[dict_idx][key].append(episode_key_data)
                agent_dicts[env_idx] = defaultdict(list)

    def save(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)

        save_paths = []
        for dict_idx, agent_idx in enumerate(self.agent_indices):
            agent_dicts = self.full_traj_dicts[dict_idx]
            dump_dict = {k: np.asarray(v) for k, v in agent_dicts.items()}

            save_path = os.path.join(save_dir, f"agent_{agent_idx}.npz")
            np.savez(save_path, **dump_dict)
            save_paths.append(save_path)

            # save win trajectory in EA
            agent_dicts = self.ea_win_trajectory[dict_idx]
            dump_dict = {k: np.asarray(v) for k, v in agent_dicts.items()}

            save_path = os.path.join(save_dir, f"agent_{agent_idx}_EA.npz")
            np.savez(save_path, **dump_dict)
            save_paths.append(save_path)

        return save_paths


def simulate(venv, policies, render=False, record=True, need_ea = False, ea_params = None,
             guide=False, guide_action=None, guide_p=0.4):
    """
    Run Environment env with the policies in `policies`.
    :param venv(VecEnv): vector environment.
    :param policies(list<BaseModel>): a policy per agent.
    :param render: (bool) true if the run should be rendered to the screen
    :param record: (bool) true if should record transparent data (if any).
    :return: streams information about the simulation
    """
    observations = venv.reset()
    dones = [False] * venv.num_envs
    states = [None for _ in policies]
    venv.policies = policies
    if need_ea:
        venv.need_ea = need_ea
        venv.ea_params = ea_params

    while True:
        if render:
            venv.render()

        actions = []
        values = []
        new_states = []
        neglogps = []

        for policy_ind, (policy, obs, state) in enumerate(zip(policies, observations, states)):
            if policy_ind == 0 and guide and random_unit(guide_p):
                act = random.choice(guide_action)
                act = act.reshape((1, len(act)))
                value, new_state, neglogp = None, None, None
            else:
                try:
                    return_tuple = policy.predict_transparent(obs, state=state, mask=dones)
                    act, value, new_state, neglogp, transparent_data = return_tuple
                    if record:
                        venv.record_extra_data(transparent_data, policy_ind)
                except AttributeError:
                    act, value, new_state, neglogp = policy.predict(obs, state=state, mask=dones)

            actions.append(act)
            values.append(value)
            new_states.append(new_state)
            neglogps.append(neglogp)

        actions = tuple(actions)
        states = new_states

        venv.values_record = values

        venv.temp_states = states
        venv.temp_observations = observations
        observations, rewards, dones, infos = venv.step(actions)
        yield observations, rewards, dones, infos


def _apply_wrappers(wrappers, multi_env):
    """Helper method to apply wrappers if they are present. Returns wrapped multi_env"""
    if wrappers is None:
        wrappers = []
    for wrap in wrappers:
        multi_env = wrap(multi_env)
    return multi_env


def make_env(
        env_name,
        seed,
        i,
        out_dir,
        our_idx=None,
        pre_wrappers=None,
        post_wrappers=None,
        agent_wrappers=None,
):
    multi_env = gym.make(env_name)

    if agent_wrappers is not None:
        for agent_id in agent_wrappers:
            multi_env.agents[agent_id] = agent_wrappers[agent_id](multi_env.agents[agent_id])

    multi_env = _apply_wrappers(pre_wrappers, multi_env)

    if not hasattr(multi_env, "num_agents"):
        multi_env = SingleToMulti(multi_env)

    if out_dir is not None:
        mon_dir = osp.join(out_dir, "mon")
        os.makedirs(mon_dir, exist_ok=True)
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, "log{}".format(i)), our_idx, allow_early_resets=False)

    multi_env = _apply_wrappers(post_wrappers, multi_env)

    multi_env.seed(seed + i)

    return multi_env
