from stable_baselines3.common.callbacks import BaseCallback

from agents.malmo_agent import MalmoAgent
from mission.mission_manager import MissionManager


class DisableMalmoAIForTrainingCallback(BaseCallback):

    def __init__(self, mission_manager: MissionManager, agent: MalmoAgent, verbose=0):
        super(DisableMalmoAIForTrainingCallback, self).__init__(verbose)
        self.manager = mission_manager
        self.agent = agent

    def _on_rollout_start(self) -> None:
        self.manager.enable_ai()
        self.agent.resume()

    def _on_rollout_end(self) -> None:
        self.manager.disable_ai()
        self.manager.agent_host.sendCommand("move 0")
        self.manager.agent_host.sendCommand("strafe 0")
        self.agent.pause()

    def _on_step(self) -> bool:
        return True
