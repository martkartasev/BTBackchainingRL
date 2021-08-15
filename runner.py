import time

from bt import conditions
from bt.back_chain_tree import BackChainTree
from observation import Observation
from world import World

MAX_DELAY = 60


class Runner:

    def __init__(self, agent):
        self.agent = agent

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