from typing import Optional, List
import logging
import csv

import numpy as np

from aintelope.agents import Environment, register_agent_class
from aintelope.agents.q_agent import QAgent, HistoryStep
from aintelope.agents.instincts.savanna_instincts import available_instincts_dict
from aintelope.environments.savanna_gym import SavannaGymEnv

logger = logging.getLogger("aintelope.agents.instinct_agent")


class InstinctAgent(QAgent):
    """Agent class with instincts"""

    def __init__(
        self,
        env: Environment,
        # model: nn.Module,
        replay_buffer: ReplayBuffer,
        warm_start_steps: int,
        target_instincts: List[str] = [],
    ) -> None:
        """
        Args:
            env (Environment): environment instance
            #model (nn.Module): neural network instance
            replay_buffer (ReplayBuffer): replay buffer of the agent
            warm_start_steps (int): amount of initial random buffer
            target_instincts (List[str]): names if used instincts
        """
        self.target_instincts = target_instincts
        self.instincts = {}
        self.done = False

        # reset after attribute setup
        super().__init__(
            env=env,
            # model=model,
            replay_buffer=replay_buffer,
            warm_start_steps=warm_start_steps,
        )

    def reset(self) -> None:
        """Reset environment and initialize instincts"""
        super().reset()
        self.init_instincts()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Decide what action to carry out using an
        epsilon-greedy policy.

        Args:
            net (nn.Module): neural network instance
            epsilon (float): value to determine likelihood of taking a random action
            device (str): current device

        Returns:
            action (int): index of action
        """
        action = super().get_action(net, epsilon, device)
        # Add further instinctual responses here later to modify action
        return action

    def update(
        self,
        env: SavannaGymEnv = None,  # TODO hack, figure out if state_to_namedtuple can be static somewhere
        observation: npt.NDArray[ObservationFloat] = None,
        score: float = 0.0,
        done: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Takes observations and updates trainer on perceived experiences. Needed here to catch instincts.

        Args:
            observation: ObservationArray
            score: Only baseline uses score as a reward
            done: boolean whether run is done

        Returns:
            None
        """
        next_state = observation
        # For future: add state (interoception) handling here when needed
        # TODO: hacky. empty next states introduced by new example code,
        # and I'm wondering if we need to save these steps too due to agent death
        # Discussion in slack.

        # interrupt to do instinctual learning
        if len(self.instincts) == 0:
            # use env reward if no instincts available
            instinct_events = []
            reward = score
        else:
            # interpret new_state and score to compute actual reward
            reward = 0
            instinct_events = []
            for instinct_name, instinct_object in self.instincts.items():
                instinct_reward, instinct_event = instinct_object.calc_reward(
                    self, new_state
                )
                reward += instinct_reward
                logger.debug(
                    f"Reward of {instinct_name}: {instinct_reward}; "
                    f"total reward: {reward}"
                )
                if instinct_event != 0:
                    instinct_events.append((instinct_name, instinct_event))
        # interruption done

        if next_state is not None:
            next_s_hist = env.state_to_namedtuple(next_state.tolist())
        else:
            next_s_hist = None
        self.history.append(
            HistoryStep(
                state=env.state_to_namedtuple(self.state.tolist()),
                action=self.last_action,
                reward=reward,
                done=done,
                instinct_events=instinct_events,
                next_state=next_s_hist,
            )
        )

        if save_path is not None:
            with open(save_path, "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [
                        self.state.tolist(),
                        self.last_action,
                        score,
                        done,
                        instinct_events,
                        next_state,
                    ]
                )

        self.trainer.update_memory(
            self.id, self.state, self.last_action, score, done, next_state
        )
        self.state = next_state

    def init_instincts(self) -> None:
        logger.debug(f"target_instincts: {self.target_instincts}")
        for instinct_name in self.target_instincts:
            if instinct_name not in available_instincts_dict:
                logger.warning(
                    f"Warning: could not find {instinct_name} in available_instincts_dict"
                )
                continue

        self.instincts = {
            instinct: available_instincts_dict.get(instinct)()
            for instinct in self.target_instincts
            if instinct in available_instincts_dict
        }
        for instinct in self.instincts.values():
            instinct.reset()


register_agent_class("instinct_agent", InstinctAgent)
