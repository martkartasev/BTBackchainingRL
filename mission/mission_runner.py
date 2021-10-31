import time

from py_trees.common import Status

from mission.mission_manager import MissionManager


class MissionRunner:

    def __init__(self, agent, active_entities=True, filename=None):
        self.mission_manager = MissionManager(agent.agent_host, filename)
        self.agent = agent
        self.active_entities = active_entities
        self.observation_manager = agent.observation_manager

    def tick_mission(self):
        host = self.mission_manager.agent_host
        observations = None
        world_state = None
        reward = 0

        while observations is None or len(observations) == 0:
            world_state = host.getWorldState()
            observations = world_state.observations
            reward += sum(reward.getValue() for reward in world_state.rewards)

        self.observation_manager.update(observations, reward)
        return world_state

    def run_mission(self):
        self.reset()

        world_state = self.tick_mission()
        while world_state.is_mission_running:
            for error in world_state.errors:
                print("Error:", error.text)

            self.agent.control_loop()

            if self.agent.is_mission_over() or self.agent.tree.status == Status.SUCCESS or self.agent.tree.status == Status.FAILURE:
                self.mission_manager.quit()
                break

            world_state = self.tick_mission()

    def run(self):
        while True:
            start = time.time()
            self.run_mission()
            end = time.time()

            print("took " + str((end - start) * 1000) + ' milliseconds')
            print("Mission ended")

    def reset(self):
        if self.mission_manager.agent_host.getWorldState().is_mission_running:
            self.mission_manager.quit()
        self.mission_manager.mission_initialization()
        self.tick_mission()
        if not self.active_entities:
            self.mission_manager.disable_ai()  # Done after tick mission to ensure that the entities have spawned
