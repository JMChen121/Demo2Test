from aprl.abs.abstracter import Abstracter, ScoreInspector


class StateAbstracter:

    def __init__(self, raw_state_dim, state_dim, state_lb, state_ub, action_dim, action_lb=-2, action_ub=2,
                 step=1, grid_num=10, epsilon=0.2, mode="state", reduction=True):

        NECSA_dict = {
            'raw_state_dim': raw_state_dim,
            'state_dim': state_dim,
            # 'state_dim': 10,
            'state_min': state_lb,
            'state_max': state_ub,
            'action_dim': action_dim,
            'action_min': action_lb,
            'action_max': action_ub,
            'step': step,
            'grid_num': grid_num,
            'epsilon': epsilon,
            'mode': mode,
            'reduction': reduction,
        }

        self.abstracter = Abstracter(NECSA_dict['step'], NECSA_dict['epsilon'])

        self.abstracter.inspector = ScoreInspector(
            NECSA_dict['step'],
            NECSA_dict['grid_num'],
            NECSA_dict['raw_state_dim'],
            NECSA_dict['state_dim'],
            NECSA_dict['state_min'],
            NECSA_dict['state_max'],
            NECSA_dict['action_dim'],
            NECSA_dict['action_min'],
            NECSA_dict['action_max'],
            NECSA_dict['mode'],
            NECSA_dict['reduction']
        )

        self.grid_dict = dict()

        self.grid_states = dict()

        self.threshold = None

    def eval_diversity(self, states):
        for episode_states in states:
            abs_states, reduction_states = self.get_state_grid_ids(episode_states)
            for idx, grid_id in enumerate(abs_states):
                if grid_id in self.grid_dict.keys():
                    self.grid_dict[grid_id] += 1
                    self.grid_states[grid_id].append(reduction_states[idx])
                else:
                    self.grid_dict[grid_id] = 1
                    self.grid_states[grid_id] = [reduction_states[idx]]
        return self.grid_dict

    def get_state_grid_ids(self, states):
        states_reduction = self.abstracter.dim_reduction(states) if self.abstracter.inspector.reduction else states
        abs_states = self.abstracter.inspector.discretize_states(states_reduction)
        return abs_states, states_reduction

