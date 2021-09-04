import json

import numpy as np
from gym.spaces import Box

CIRCLE_DEGREES = 360

RELATIVE_DISTANCE_AXIS_MAX = 1000

ENEMY_TYPE = "VindicationIllager"

# TODO: This shouldn't be hard-coded
GRID_SIZE = 3 * 3 * 2
GRID_SIZE_AXIS = [3, 2, 3]

# Always append at the end of this list
game_objects = ["dirt", "grass", "stone", "fire", "air"]


def get_skeleton_info(info):
    if "Entities" in info:
        for entity in info["Entities"]:
            if entity.get("name") == ENEMY_TYPE:
                return entity


def get_yaw(info):
    if "Yaw" in info:
        yaw = info["Yaw"]
        if yaw <= 0:
            yaw += CIRCLE_DEGREES
        return yaw
    return None


def get_pitch(info):
    return info.get("Pitch")


def get_direction_vector(info):
    yaw = get_yaw(info)
    pitch = get_pitch(info)

    if yaw is None or pitch is None:
        return np.zeros(3)

    yaw_radians = degrees_to_radians(yaw)
    pitch_radians = degrees_to_radians(pitch)

    direction_vector = np.array([
        -np.sin(yaw_radians) * np.cos(pitch_radians),
        -np.sin(pitch_radians),
        np.cos(yaw_radians) * np.cos(pitch_radians),
    ])
    return direction_vector


def degrees_to_radians(deg):
    half_circle = CIRCLE_DEGREES / 2
    return deg * np.pi / half_circle


def get_grid_obs_vector(grid):
    grid_ordinals = [get_game_object_ordinal(block) for block in grid]
    return np.array(grid_ordinals, dtype=np.float32)


def get_game_object_ordinal(game_object):
    if game_object is None:
        return -1
    if game_object not in game_objects:
        print(f"Object {game_object} has not been added to the game objects list")
        return -1
    else:
        return game_objects.index(game_object)


class Observation:

    def __init__(self, observations):
        if observations is None or len(observations) == 0:
            print("Observations is null or empty")
            return

        info_json = observations[-1].text
        if info_json is None:
            print("Info is null")
            return

        info = json.loads(info_json)

        skeleton_info = get_skeleton_info(info)

        if skeleton_info is not None:
            skeleton_position_list = [skeleton_info.get("x"), skeleton_info.get("y"), skeleton_info.get("z")]
        else:
            skeleton_position_list = [None, None, None]

        skeleton_position = None if None in skeleton_position_list else np.array(skeleton_position_list)
        player_position_list = [info.get("XPos"), info.get("YPos"), info.get("ZPos")]
        player_position = None if None in player_position_list else np.array(player_position_list)


        if player_position is not None and skeleton_position is not None:
            relative_position = skeleton_position - player_position
            relative_position = np.clip(relative_position, -RELATIVE_DISTANCE_AXIS_MAX, RELATIVE_DISTANCE_AXIS_MAX)
        else:
            relative_position = np.zeros(3)

        direction_vector = get_direction_vector(info)

        player_life = info.get("Life", 0)

        surroundings_list = info.get("Surroundings")

        if surroundings_list is not None:
            surroundings = get_grid_obs_vector(surroundings_list)
        else:
            surroundings = -1 * np.ones(GRID_SIZE)

        self.vector = np.hstack((
            relative_position,
            direction_vector,
            player_life,
            surroundings
        ))

    @staticmethod
    def get_observation_space():
        low_position = -RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)
        low_direction = -1 * np.ones(3)
        low_player_life = 0
        low_surroundings = -1 * np.ones(GRID_SIZE)
        low = np.hstack((
            low_position,
            low_direction,
            low_player_life,
            low_surroundings
        ))

        high_position = RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)
        high_direction = np.ones(3)
        high_player_life = 100
        high_surroundings = len(game_objects) * np.ones(GRID_SIZE)
        high = np.hstack((
            high_position,
            high_direction,
            high_player_life,
            high_surroundings
        ))

        return Box(low, high)
