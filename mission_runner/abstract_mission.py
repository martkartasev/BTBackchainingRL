import os
import sys
import time
from builtins import range

from malmo import MalmoPython

from observation import Observation

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools

    print = functools.partial(print, flush=True)


class AbstractMission:
    def __init__(self, agent_host, filename=None):
        self.agent = agent_host
        self.agentExited = False
        self.timesteps = 0
        if filename is not None:
            if isinstance(filename, list):
                missions = list()
                records = list()
                for name in filename:
                    mission, record = self.create_mission(self.read_mission(name))
                    missions.append(mission)
                    records.append(record)
                self.mission = missions
                self.mission_record = records
            else:
                self.mission, self.mission_record = self.create_mission(self.read_mission(filename))
        self.counter = 0

    def read_mission(self, filename):
        with open(filename, 'r') as myfile:
            data = myfile.read().replace('\n', '')
        return data

    def create_mission(self, missionString=None):
        if missionString is not None:
            mission = MalmoPython.MissionSpec(missionString, True)
        else:
            mission = MalmoPython.MissionSpec()

        record = MalmoPython.MissionRecordSpec()
        #  mission.forceWorldReset()
        mission.allowAllInventoryCommands()
        # mission.observeFullInventory()

        return mission, record

    def mission_initialization(self):
        # Attempt to start a mission:
        self.agentExited = False
        self.agent.reset_agent()
        max_retries = 25
        for retry in range(max_retries):
            try:
                if isinstance(self.mission, list):
                    i = self.counter % len(self.mission)
                    self.agent.startMission(self.mission[i], self.mission_record[i])
                    self.counter = self.counter + 1
                else:
                    self.agent.start_mission(self.mission, self.mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    print("Failed to connect to mission, retrying after 5 seconds: ", e)
                    time.sleep(5)

        print("Waiting for the mission to start ", end=' ')
        world_state = self.agent.get_world_state()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent.get_world_state()
            for error in world_state.errors:
                print("Error:", error.text)

        print()
        print("Running mission", end=' ')

        self.agent.activate_night_vision()
        self.agent.set_fire_eternal()

        return world_state

    # Wait for valid state instead...
    def soft_reset(self):
        self.agent.go_to_spawn()
        self.agent.destroy_all_entities()
        self.wait_for_entity(False)
        self.agent.get_next_observations_and_reward()
        self.agent.create_static_skeleton()
        self.agent.create_cow()
        self.wait_for_entity(True)
        self.agent.get_next_observations_and_reward()

    def wait_for_entity(self, expect_entity):
        while True:
            observations, _ = self.agent.get_next_observations_and_reward()
            observation = Observation(observations)
            if expect_entity and observation.dict["entity_visible"] == 1:
                break
            elif not expect_entity and observation.dict["entity_visible"] == 0:
                break

    def run_mission(self):
        raise NotImplementedError()
