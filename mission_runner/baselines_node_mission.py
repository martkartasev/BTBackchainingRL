import time

from mission_runner.mission_manager import MissionManager


class BaselinesNodeMission:

    def __init__(self, agent, tree, filename=None, hard_reset=True):
        self.mission_manager = MissionManager(agent.agent_host, filename)
        self.agent = agent
        self.observation_manager = agent.observation_manager
        self.hard_reset = hard_reset
        self.tree = tree

    def tick_mission(self):
        host = self.mission_manager.agent_host
        observations = None
        world_state = None
        reward = 0

        while observations is None or len(observations) == 0:
            world_state = host.get_world_state()
            observations = world_state.observations
            reward = sum(reward.getValue() for reward in world_state.rewards)

        self.observation_manager.update(observations, reward)
        return world_state

    def run_mission(self):
        world_state = self.tick_mission()
        while world_state.is_mission_running:
            for error in world_state.errors:
                print("Error:", error.text)

            if self.agent.is_mission_over():
                self.mission_manager.quit()
                break

            self.agent.control_loop()

            world_state = self.tick_mission()

    def run(self):
        while True:
            self.mission_manager.mission_initialization()
            if not self.hard_reset:
                self.soft_reset()

            start = time.time()
            self.run_mission()
            end = time.time()

            print("took " + str((end - start) * 1000) + ' milliseconds')
            print("Mission ended")

    def soft_reset(self):
        self.mission_manager.go_to_spawn()
        self.mission_manager.destroy_all_entities()
        self.wait_for_entity(False)

        self.mission_manager.create_static_skeleton()
        self.mission_manager.create_cow()
        self.wait_for_entity(True)

    def wait_for_entity(self, expect_entity):
        while True:
            observation, _ = self.tick_mission()
            if expect_entity == observation.dict["entity_visible"]:
                break
