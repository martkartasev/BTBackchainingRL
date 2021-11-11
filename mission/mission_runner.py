import time

from mission.mission_manager import MissionManager
from py_trees.common import Status


class MissionRunner:

    def __init__(self, agent, active_entities=True, filename=None, evaluation_manager=None):
        self.mission_manager = MissionManager(agent.agent_host, filename)
        self.agent = agent
        self.active_entities = active_entities
        self.observation_manager = agent.observation_manager
        self.evaluation_manager = evaluation_manager

    def run(self):
        mission = 0
        while True:
            mission += 1
            start = time.time()
            if self.evaluation_manager is not None:
                self.evaluation_manager.record_mission_start(start)

            state, steps = self.run_mission()

            end = time.time()

            print("Took " + str((end - start) * 1000) + ' milliseconds')
            print("Mission " + str(mission) + " ended")

            if self.evaluation_manager is not None:
                self.evaluation_manager.record_mission_end(self.agent.is_mission_over(), steps, end)
                if self.evaluation_manager.runs <= mission:
                    self.evaluation_manager.store_evaluation()
                    break

    def run_mission(self):
        self.reset()
        steps = 0
        world_state = self.tick_mission()
        while world_state.is_mission_running:
            for error in world_state.errors:
                print("Error:", error.text)

            self.agent.control_loop()
            steps += 1

            tree_status = self.agent.tree.status
            if self.agent.is_mission_over() or tree_status == Status.SUCCESS or tree_status == Status.FAILURE:
                self.mission_manager.quit()
                break

            world_state = self.tick_mission()
            if self.evaluation_manager is not None:
                position = self.observation_manager.get_position()
                self.evaluation_manager.record_position(position[0], position[2])

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
        self.mission_manager.mission_initialization()
        self.tick_mission()
        if not self.active_entities:
            self.mission_manager.disable_ai()  # Done after tick mission to ensure that the entities have spawned
