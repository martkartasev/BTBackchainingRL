import time

from py_trees.common import Status

from mission.mission_manager import MissionManager


class MissionRunner:

    def __init__(self, agent, active_entities=True, filename=None, evaluation_manager=None, mission_max_time=None, logging=0):
        self.mission_manager = MissionManager(agent.agent_host, filename, logging)
        self.agent = agent
        self.active_entities = active_entities
        self.observation_manager = agent.observation_manager
        self.evaluation_manager = evaluation_manager
        self.mission_max_time = mission_max_time
        self.mission_start_time = time.time()
        self.logging = logging

    def run(self):
        mission = 0
        while True:
            mission += 1
            self.mission_start_time = time.time()
            if self.evaluation_manager is not None:
                self.evaluation_manager.record_mission_start(self.mission_start_time)

            state, steps = self.run_mission()

            end = time.time()
            if self.logging > 0:
                print("Took " + str((end - self.mission_start_time) * 1000) + ' milliseconds')
                print("Mission " + str(mission) + " ended")

            if self.evaluation_manager is not None:
                self.evaluation_manager.record_mission_end(self.agent.is_mission_over(), steps, end)
                if self.evaluation_manager.runs <= mission:
                    self.evaluation_manager.store_evaluation()
                    break

    def run_mission(self):
        self.reset()
        self.mission_start_time = time.time()
        steps = 0
        world_state = self.tick_mission()
        while world_state.is_mission_running:
            for error in world_state.errors:
                print("Error:", error.text)

            self.agent.control_loop()
            steps += 1

            if self.evaluation_manager is not None:
                position = self.observation_manager.get_position()
                self.evaluation_manager.record_position(position[0], position[2])

            tree_status = self.agent.tree.status
            tree_finished = (tree_status == Status.SUCCESS or tree_status == Status.FAILURE)
            timed_out = self.mission_max_time is not None and time.time() - self.mission_start_time >= self.mission_max_time
            if self.agent.is_mission_over() or tree_finished or timed_out:
                self.mission_manager.quit()
                break

            world_state = self.tick_mission()

        return world_state, steps

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

    def reset(self):
        if self.mission_manager.agent_host.getWorldState().is_mission_running:
            self.mission_manager.quit()

        done = False
        while not done:
            self.mission_manager.mission_initialization()
            tries = 0

            while not done and tries < 100:  # Sync mission and client
                self.tick_mission()
                done = not self.agent.is_mission_over()

                tries += 1

            if not self.active_entities:
                self.mission_manager.disable_ai()  # Done after tick mission to ensure that the entities have spawned
