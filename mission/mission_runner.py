import time

from mission.mission_manager import MissionManager


class MissionRunner:

    def __init__(self, agent, filename=None, hard_reset=True):
        self.mission_manager = MissionManager(agent.agent_host, filename)
        self.agent = agent
        self.observation_manager = agent.observation_manager
        self.hard_reset = hard_reset

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

            if self.agent.is_mission_over():
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
        if self.hard_reset:
            if self.mission_manager.agent_host.getWorldState().is_mission_running:
                self.mission_manager.quit()
            self.mission_manager.mission_initialization()
            self.tick_mission()
        else:
            if self.mission_manager.agent_host.getWorldState().is_mission_running:
                self.soft_reset()
            else:
                self.mission_manager.mission_initialization()
                self.tick_mission()
                self.soft_reset()

    def soft_reset(self):
        self.mission_manager.go_to_spawn()
        self.mission_manager.destroy_all_entities()
        self.wait_for_entity(False)

        self.mission_manager.create_static_skeleton()
        self.mission_manager.create_cow()
        self.wait_for_entity(True)

    def wait_for_entity(self, expect_entity):
        while True:
            self.tick_mission()
            observation = self.observation_manager.observation
            if expect_entity == observation.dict["entity_visible"]:
                break
