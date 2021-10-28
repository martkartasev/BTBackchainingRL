from py_trees.common import Status

from agents.malmo_agent import MalmoAgent


class BehaviorTreeAgent(MalmoAgent):

    def __init__(self, agent_host, observation_manager, name="execution/node_skeleton"):
        super().__init__(agent_host)
        self.name = name
        self.observation_manager = observation_manager
        self.tree = None

    def reset(self):
        self.observation_manager.reset()

    def is_mission_over(self):
        return not self.observation_manager.is_agent_alive() or self.tree.status == Status.SUCCESS or self.tree.status == Status.FAILURE

    def control_loop(self):
        self.tree.tick_once()
