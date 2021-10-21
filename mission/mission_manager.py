import os
import sys
import time
from builtins import range

try:
    from malmo import MalmoPython
except ImportError:
    import MalmoPython

import numpy

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


MAX_RETRIES = 25
WAIT_FOR_SECONDS = 10
WAIT_INTERVAL = 0.1


class MissionManager:
    def __init__(self, agent_host: MalmoPython.AgentHost, filename=None):
        self.agent_host = agent_host
        self.counter = 0

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

    def mission_initialization(self):
        # Attempt to start a mission:
        max_retries = MAX_RETRIES

        world_state = None
        for retry in range(MAX_RETRIES):
            try:
                if isinstance(self.mission, list):
                    i = self.counter % len(self.mission)
                    self.agent_host.startMission(self.mission[i], self.mission_record[i])
                    self.counter = self.counter + 1
                else:
                    self.agent_host.startMission(self.mission, self.mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    print("Failed to connect to mission, retrying after 500 milliseconds: ", e)
                    time.sleep(0.5)

            print("Waiting for the mission to start ", end=' ')
            world_state = self.agent_host.getWorldState()

            wait_counter = 0
            while not world_state.has_mission_begun:
                print(".", end="")
                wait_counter += 1
                time.sleep(WAIT_INTERVAL)
                world_state = self.agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)

                if wait_counter > WAIT_FOR_SECONDS / WAIT_INTERVAL:
                    break

        print()
        print("Running mission", end=' ')

        self.activate_night_vision()
        self.set_fire_eternal()
        self.make_hungry()

        return world_state

    def destroy_all_entities(self):
        self.agent_host.sendCommand("chat /kill @e[type=skeleton]")
        self.agent_host.sendCommand("chat /kill @e[type=cow]")
        time.sleep(0.01)
        self.agent_host.sendCommand("chat /kill @e[type=item]")  # Do this last to remove drops from the mobs

    def go_to_spawn(self, vec=numpy.array([0, 4, 0])):
        self.agent_host.sendCommand("chat /tp " + " ".join(vec.astype(str)))

    def activate_night_vision(self):
        self.agent_host.sendCommand("chat /effect @p night_vision 99999 255")

    def set_fire_eternal(self):
        self.agent_host.sendCommand("chat /gamerule doFireTick false")

    def make_hungry(self):
        self.agent_host.sendCommand("chat /effect @p hunger 5 255")

    def create_static_skeleton(self):
        self.agent_host.sendCommand("chat /summon skeleton -14 4 0 {NoAI:1}")

    def create_cow(self):
        self.agent_host.sendCommand("chat /summon cow 14 4 0 {NoAI:1}")

    def quit(self):
        self.agent_host.sendCommand("quit")
