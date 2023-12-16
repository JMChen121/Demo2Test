import numpy as np


def convert_save(base_path, agent=0):
    data = np.load(base_path + f"agent_{agent}.npz", allow_pickle=True)
    # data_win = np.load(base_path + f"agent_{agent}_win.npz", allow_pickle=True)
    s = np.load(base_path + "score.npy", allow_pickle=True).item()
    wins_idx = [i for i, x in enumerate(s["winners"]) if x == agent]

    if len(wins_idx) > 100:
        wins_idx = wins_idx[0:100]

    dump_dict = {k: np.asarray(data[k][wins_idx]) for k in data.files[:-1]}
    dump_dict[data.files[-1]] = data[data.files[-1]]

    np.savez(base_path + f"agent_{agent}_win.npz", **dump_dict)
    # np.save(base_path + f"agent_{agent}_win.npz", data)


# base = 'data/trajectories/2023-10-12-08-47-08_RunToGoalAnts_v1-v1_ep-200_ea-True/'
# base = 'data/trajectories/2023-10-12-23-49-38_KickAndDefend_v3-v3_ep-200_ea-True/'
# base = 'data/trajectories/2023-10-03-05-32-56_RunToGoalHumans/'
# base = 'data/trajectories/2023-10-09-10-56-25_SumoAnts_v1-v1_ep-500_ea-False/'

# base = 'data/trajectories/2023-09-09-17-32-57_YouShallNotPassHumans/'
# base = 'data/trajectories/2023-09-09-16-11-14_SumoHumans/'
# base = 'data/trajectories/2023-09-09-18-36-34_KickAndDefend/'

agent_idx = 1
convert_save(base, agent_idx)

