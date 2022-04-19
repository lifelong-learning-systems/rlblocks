import typing

import tella


class RandAgent(tella.ContinualRLAgent):
    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        return [
            self.action_space.sample() if obs is not None else None
            for obs in observations
        ]

    def receive_transitions(
        self, transitions: typing.List[typing.Optional[tella.Transition]]
    ) -> None:
        pass
