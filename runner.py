import time

from bt import conditions
from bt.back_chain_tree import BackChainTree
from observation import Observation
from utils.strings import tree_to_string
from world import World

MAX_DELAY = 60

#TODO: Delete
class Runner:

    def __init__(self, agent):
        self.agent = agent
        self.agentExited = False

        self.night_vision = True

        self.world = World(self.agent)

        self.last_delta = time.time()

        self.tree = BackChainTree(agent, conditions.IsNotInFire(agent))

        self.world.start_world()

    def run_mission(self):
        self.last_delta = time.time()

        world_state = self.agent.get_next_world_state()

        if self.night_vision:
            self.agent.activate_night_vision()

        while world_state is not None and world_state.is_mission_running:
            observation = Observation(self.agent.get_next_world_state().observations)
            self.agent.set_observation(observation)

            self.tree.root.tick_once()

            print(tree_to_string(self.tree.root))

            self.check_timeout(self.world, world_state)

    def check_timeout(self, world, world_state):
        if (world_state.number_of_video_frames_since_last_state > 0 or
                world_state.number_of_observations_since_last_state > 0 or
                world_state.number_of_rewards_since_last_state > 0):
            self.last_delta = time.time()
        else:
            if time.time() - self.last_delta > MAX_DELAY:
                print("Max delay exceeded for world state change")
                world.restart_minecraft(world_state, "world state change")

    def mission_initialization(self):
        # Attempt to start a mission:
        self.agentExited = False
        self.agent.reset_agent()
        max_retries = 8
        for retry in range(max_retries):
            try:
                if isinstance(self.mission, list):
                    i = self.counter % len(self.mission)
                    self.agent.startMission(self.mission[i], self.mission_record[i])
                    self.counter = self.counter + 1
                else:
                    self.agent.startMission(self.mission, self.mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    print("Failed to connect to mission, retrying after 5 seconds: ", e)
                    time.sleep(5)

        print("Waiting for the mission to start ", end=' ')
        world_state = self.agent.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

        print()
        print("Running mission", end=' ')
        return world_state


    def run(self):
        self.mission_initialization()

        start = time.time()
        self.run_mission()
        end = time.time()

        print("took " + str((end - start) * 1000) + ' milliseconds')
        print("Mission ended")

