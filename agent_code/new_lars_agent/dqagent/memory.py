
import numpy as np

from ..utils import returns as ret, all_equal


class EpisodeMemory:
    def __init__(self, observations=None, actions=None, rewards=None, returns=None):
        if observations is None:
            observations = []
        if actions is None:
            actions = []
        if rewards is None:
            rewards = []
        if returns is None:
            returns = []

        self.observations = list(observations)
        self.actions = list(actions)
        self.rewards = list(rewards)
        self.returns = list(returns)

    def add(self, observation, action, reward, r=None):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        if r is not None:
            self.returns.append(r)

    def finalize(self, gamma):
        # finalize an episode by calculating expected reward in every step
        self.returns = ret(self.rewards, gamma)
        # the total discounted reward for the episode is at index 0
        return self.returns[0]

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.returns = []

    def reduce(self, size):
        if size >= len(self):
            return

        self.observations = self.observations[-size:]
        self.actions = self.actions[-size:]
        self.rewards = self.rewards[-size:]
        self.returns = self.returns[-size:]

    def is_complete(self):
        return all_equal(len(self.observations), len(self.actions), len(self.rewards), len(self.returns))

    def __len__(self):
        if not self.is_complete():
            raise RuntimeError("Cannot define length on incomplete Memory.")

        return len(self.observations)


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
        discounted_reward = list(map(lambda e: e.finalize(gamma), self.episodes))[0]

        return discounted_reward

    def aggregate(self):
        for episode in self.episodes:
            if not episode.is_complete():
                raise RuntimeError("Cannot Aggregate incomplete EpisodeMemory's into Memory.")

        for episode in self.episodes:
            self.observations.extend(episode.observations)
            self.actions.extend(episode.actions)
            self.rewards.extend(episode.rewards)
            self.returns.extend(episode.returns)

        self.episodes = []

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
            np.asarray(self.observations)[idx],
            np.asarray(self.actions)[idx],
            np.asarray(self.rewards)[idx],
            np.asarray(self.returns)[idx]
        )

        return result

    def clear(self):
        self.episodes = []
        super().clear()

    def is_complete(self):
        if self.episodes:
            return False

        return super().is_complete()
