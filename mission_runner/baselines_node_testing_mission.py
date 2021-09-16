import time

from mission_runner.baselines_node_mission import BaselinesNodeMission


class BaselinesNodeTestingMission(BaselinesNodeMission):

    def __init__(self, agent, tree, filename=None, hard_reset=True):
        super().__init__(agent, tree, filename, hard_reset)

    def run_mission(self):
        world_state = self.agent.get_world_state()
        while world_state.is_mission_running:
            self.agent.update_observations_and_reward()
            if self.agent.is_mission_over():
                self.agent.quit()
                break

            self.tree.tick_once()

    def run(self):
        self.mission_initialization()
        if not self.hard_reset:
            self.soft_reset()

        start = time.time()
        self.run_mission()
        end = time.time()

        print("took " + str((end - start) * 1000) + ' milliseconds')
        print("Mission ended")
