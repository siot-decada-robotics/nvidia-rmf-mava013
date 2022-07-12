from mava.utils.jax_tree_utils import stack_trees, index_stacked_tree


class VecEnvWrapper:  # TODO dummy for now should probably use stable baselines SupprocVecEnv
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self):
        outputs = []
        for i in range(self.num_envs):
            outputs.append(self.envs[i].reset())
        return stack_trees(outputs)

    def step(self, actions: [int]):
        outputs = []
        for i in range(self.num_envs):
            action = index_stacked_tree(actions, i)
            outputs.append(self.envs[i].step(action))
        return stack_trees(outputs)

    def reward_spec(self):
        return self.envs[0].reward_spec()

    @property
    def possible_agents(self):
        return self.envs[0].possible_agents
