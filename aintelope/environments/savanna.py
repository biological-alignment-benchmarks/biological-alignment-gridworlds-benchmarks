import typing as typ
import functools

import numpy as np
from gym.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector

NUM_ITERS = 500  # duration of the game
MAP_DIM = 100
CYCLIC_BOUNDARIES = True
AMOUNT_AGENTS = 1  # for now only one agent
AMOUNT_GRASS = 2
OBSERVATION_SPACE = Box(0,
                        MAP_DIM,
                        shape=(2 * (AMOUNT_AGENTS + AMOUNT_GRASS), ))
ACTION_SPACE = Discrete(4)  # agent can walk in 4 directions
ACTION_MAP = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])


class RawEnv(AECEnv):

    metadata = {'name': 'savanna_v1'}

    def __init__(self):
        self.possible_agents = [
            'player_' + str(r) for r in range(AMOUNT_AGENTS)
        ]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(AMOUNT_AGENTS))))

        self._action_spaces = {
            agent: ACTION_SPACE
            for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: OBSERVATION_SPACE
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return OBSERVATION_SPACE

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return ACTION_SPACE

    def observe(self, agent: str):
        """Return observation of given agent.
        """
        return np.array(self.observations[agent])

    def render(self, mode='human'):
        """Render the environment.
        """
        raise NotImplementedError

    def close(self):
        """Release any graphical display, subprocesses, network connections
        or any other environment data which should not be kept around after
        the user is no longer using the environment.
        """
        raise NotImplementedError

    def reset(self, seed: typ.Optional[int] = None):
        """Reset needs to initialize the following attributes:
            - agents
            - rewards
            - _cumulative_rewards
            - dones
            - infos
            - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        """

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.grass = np.random.randint(0, MAP_DIM, 2 * AMOUNT_GRASS)
        self.state = {
            agent: np.random.randint(0, MAP_DIM, 2 * AMOUNT_AGENTS)
            for agent in self.agents
        }
        self.observations = {
            agent: np.concatenate([self.state[agent], self.grass])
            for agent in self.agents
        }
        self.num_moves = 0

        # cycle through the agents
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action: int):
        """Take in an action for the current agent (specified by
        agent_selection) and needs to update:
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            """
            handles stepping an agent which is already done
            accepts a None action for the one agent, and moves the
            agent_selection to the next done agent, or if there are no more
            done agents, to the next live agent
            """
            return self._was_done_step(action)

        agent = self.agent_selection
        """
        the agent which stepped last had its _cumulative_rewards accounted for
        (because it was returned by last()), so the _cumulative_rewards for
        this agent should start again at 0
        """
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[self.agents[0]], self.rewards[
                self.agents[1]] = REWARD_MAP[(self.state[self.agents[0]],
                                              self.state[self.agents[1]])]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {
                agent: self.num_moves >= NUM_ITERS
                for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[self.agents[
                    1 - self.agent_name_mapping[i]]]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


def env():
    """Add PettingZoo wrappers to environment class.
    """
    env = RawEnv()
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
