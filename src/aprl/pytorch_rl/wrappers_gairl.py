import os
import warnings
from os import path as osp

import gym
import numpy as np

from aprl.common.multi_monitor import MultiMonitor
from aprl.envs.multi_agent import SingleToMulti


def _filter_dict(d, keys):
    """Filter a dictionary to contain only the specified keys.

    If keys is None, it returns the dictionary verbatim.
    If a key in keys is not present in the dictionary, it gives a warning, but does not fail.

    param d: (dict)
    param keys: (iterable) the desired set of keys; if None, performs no filtering.
    return (dict) a filtered dictionary."""
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
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, "log{}".format(i)), our_idx)

        out_dir = osp.join(out_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

    multi_env = _apply_wrappers(post_wrappers, multi_env)

    multi_env.seed(seed + i)

    # if env_name == 'multicomp/RunToGoalAnts-v0':
    #     pass

    return multi_env
