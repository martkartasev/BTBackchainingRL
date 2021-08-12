import uuid
import time

from malmo import malmoutils
from MalmoPython import MissionSpec, ClientInfo, ClientPool

TIME_LIMIT_IN_SECONDS = 30
VIDEO_SIZE = (800, 600)

RECORDING_NAME = "saved_data"

IP = "127.0.0.1"
PORT = 10000

MAX_RETRIES = 3
MAX_RESPONSE_TIME = 60


def setup_mission(mission_data):
    mission = MissionSpec(mission_data, True)
    mission.timeLimitInSeconds(TIME_LIMIT_IN_SECONDS)
    mission.requestVideo(*VIDEO_SIZE)
    return mission


def setup_mission_record(agent):
    mission_record = malmoutils.get_default_recording_object(agent.agent_host, RECORDING_NAME)
    return mission_record


def setup_pool(client_info):
    pool = ClientPool()
    pool.add(client_info)
    return pool


def setup_experiment_id():
    experiment_id = str(uuid.uuid1())
    print(f"experiment id {experiment_id}")
    return experiment_id


def read_mission(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', '')
    return data


class World:

    def __init__(self, agent):

        self.agent = agent

        malmoutils.parse_command_line(agent.agent_host, [''])

        mission_data = read_mission('resources/arena_skeleton.xml')
        self.mission = setup_mission(mission_data)

        self.mission_record = setup_mission_record(agent)
        self.client_info = ClientInfo(IP, PORT)
        self.pool = setup_pool(self.client_info)
        self.experiment_id = setup_experiment_id()

    def start_world(self):
        for retry in range(MAX_RETRIES):
            try:
                self.agent.start_mission(self.mission, self.pool, self.mission_record, self.experiment_id)
                break
            except RuntimeError as e:
                if retry == MAX_RETRIES - 1:
                    print("Error starting world:", e)
                    exit(1)
                else:
                    time.sleep(2)

        print("Waiting for the world to start", end=' ')
        start_time = time.time()
        world_state = self.agent.get_world_state()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            if time.time() - start_time > MAX_RESPONSE_TIME:
                print("Max delay exceeded for world to begin")
                self.agent.restart_minecraft(world_state, "begin world")
            world_state = self.agent.get_world_state()
            for error in world_state.errors:
                print("Error:", error.text)
        print()
