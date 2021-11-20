import os
import sys
import time
from builtins import range

from utils.random import get_random_in_range

try:
    import MalmoPython
except ImportError:
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

    def start_mission(self):
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
                    print("Failed to connect to mission, retrying after 100 milliseconds: ", e)
                    time.sleep(WAIT_INTERVAL)

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

        return world_state

    def enable_ai(self):
        self.agent_host.sendCommand("chat /entitydata @e {NoAI:0}")

    def disable_ai(self):
        self.agent_host.sendCommand("chat /entitydata @e {NoAI:1}")

    def activate_night_vision(self):
        self.agent_host.sendCommand("chat /effect @p night_vision 99999 255")

    def set_fire_eternal(self):
        self.agent_host.sendCommand("chat /gamerule doFireTick false")

    def make_hungry(self):
        self.agent_host.sendCommand("chat /effect @p hunger 5 255")

    def quit(self):
        self.agent_host.sendCommand("quit")

    # This needs to be called before starting the mission
    def randomize_start_position(self, ranges):
        if ranges:
            x = get_random_in_range(ranges['x'])
            y = get_random_in_range(ranges['y'])
            z = get_random_in_range(ranges['z'])
            print(f"Set start position to {x} {y} {z}.")
            self.mission.startAt(x, y, z)

    # This needs to be called after starting the mission
    def randomize_entity_positions(self, entity_ranges):
        if entity_ranges:
            for entity, position_range in entity_ranges.items():
                self.randomize_entity_position(position_range, entity)

    def randomize_entity_position(self, ranges, entity_type):
        x = get_random_in_range(ranges['x'])
        y = get_random_in_range(ranges['y'])
        z = get_random_in_range(ranges['z'])
        print(f"Set {entity_type} position to {x} {y} {z}.")
        self.agent_host.sendCommand(f"chat /teleport @e[type={entity_type}] {x} {y} {z}")
