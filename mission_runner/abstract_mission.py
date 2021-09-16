import os
import sys
import time
from builtins import range

from malmo import MalmoPython

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools

    print = functools.partial(print, flush=True)


def read_mission(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', '')
    return data


def create_mission(mission_string=None):
    if mission_string is not None:
        mission = MalmoPython.MissionSpec(mission_string, True)
    else:
        mission = MalmoPython.MissionSpec()

    record = MalmoPython.MissionRecordSpec()
    #  mission.forceWorldReset()
    mission.allowAllInventoryCommands()
    # mission.observeFullInventory()

    return mission, record


class AbstractMission:
    def __init__(self, agent_host, filename=None):
        self.agent = agent_host
        self.agent_exited = False
        self.time_steps = 0
        if filename is not None:
            if isinstance(filename, list):
                missions = list()
                records = list()
                for name in filename:
                    mission, record = create_mission(read_mission(name))
                    missions.append(mission)
                    records.append(record)
                self.mission = missions
                self.mission_record = records
            else:
                self.mission, self.mission_record = create_mission(read_mission(filename))
        self.counter = 0

    def mission_initialization(self):
        # Attempt to start a mission:
        self.agent_exited = False
        self.agent.reset()
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
        self.agent.make_hungry()

        return world_state
