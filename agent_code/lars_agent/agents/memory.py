
import numpy as np

from ..utils import returns as ret, all_equal


class EpisodeMemory:
    def __init__(self, states=None, actions=None, rewards=None, new_states=None, returns=None):
        if states is None:
            states = []
        if actions is None:
            actions = []
        if rewards is None:
            rewards = []
        if new_states is None:
            new_states = []
        if returns is None:
            returns = []

        self.states = list(states)
        self.actions = list(actions)
        self.rewards = list(rewards)
        self.new_states = list(new_states)
        self.returns = list(returns)

    def add(self, state, action, reward, new_state, r=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        if r is not None:
            self.returns.append(r)

    def finalize(self, gamma):
        # finalize an episode by calculating expected reward in every step
        self.returns = ret(self.rewards, gamma)
        # the total discounted reward for the episode is at index 0
        return self.returns[0]

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.returns = []

    def reduce(self, size):
        if size >= len(self):
            return

        self.states = self.states[-size:]
        self.actions = self.actions[-size:]
        self.rewards = self.rewards[-size:]
        self.new_states = self.new_states[-size:]
        self.returns = self.returns[-size:]

    def _check_complete(self):
        if not all_equal(len(self.states), len(self.actions), len(self.rewards), len(self.new_states)):
            raise RuntimeError("Cannot define length on incomplete Memory.")

    def __len__(self):
        self._check_complete()

        return len(self.states)


class Memory(EpisodeMemory):
    def __init__(self, episodes=None, *args, **kwargs):
        if episodes is None:
            episodes = []

        self.episodes = list(episodes)

        super().__init__(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.episodes[-1].add(*args, **kwargs)

    def add_episode(self, *args, **kwargs):
        self.episodes.append(EpisodeMemory(*args, **kwargs))

    def finalize(self, gamma):
        # finalize all episodes and return their sum
        reward = list(map(lambda e: e.finalize(gamma), self.episodes))[0]

        return reward

    def aggregate(self):
        for episode in self.episodes:
            self.states.extend(episode.states)
            self.actions.extend(episode.actions)
            self.rewards.extend(episode.rewards)
            self.new_states.extend(episode.new_states)
            self.returns.extend(episode.returns)

        self.episodes = []

        self._check_complete()

    def random_batch(self, size, reduce_ok=True, keep_episodes=False):
        self.aggregate()

        if size >= len(self):
            if not reduce_ok:
                raise RuntimeError(f"Random Batch of size {size} requested on Memory of size {len(self)} with reduce_ok=False.")
            idx = np.arange(len(self))
            np.random.shuffle(idx)
        else:
            idx = np.random.choice(np.arange(len(self)), size, replace=False)

        result = Memory(
            self.episodes if keep_episodes else None,
            np.asarray(self.states)[idx],
            np.asarray(self.actions)[idx],
            np.asarray(self.rewards)[idx],
            np.asarray(self.new_states)[idx],
            np.asarray(self.returns)[idx]
        )

        return result

    def clear(self):
        self.episodes = []
        super().clear()

    def _check_complete(self):
        if self.episodes:
            raise RuntimeError("Memory with un-aggregated episodes is incomplete.")

        return super()._check_complete()
