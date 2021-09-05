import json

import numpy as np
from gym.spaces import Box

CIRCLE_DEGREES = 360

ARENA_SIZE = 16
RELATIVE_DISTANCE_AXIS_MAX = 50

PLAYER_MAX_LIFE = 100

ENEMY_TYPE = "Skeleton"
ENEMY_MAX_LIFE = 24

# TODO: This shouldn't be hard-coded
GRID_SIZE = 3 * 3 * 2
GRID_SIZE_AXIS = [3, 2, 3]

# Always append at the end of this list
game_objects = ["dirt", "grass", "stone", "fire", "air", "brick_block"]


def get_entity_info(info, entity_name):
    if "Entities" in info:
        for entity in info["Entities"]:
            if entity.get("name") == entity_name:
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
        return 0
    if game_object not in game_objects:
        print(f"Object {game_object} has not been added to the game objects list")
        return 0
    else:
        return game_objects.index(game_object) + 1


def get_standardized_relative_position(skeleton_info, player_position):
    if skeleton_info is not None:
        skeleton_position_list = [skeleton_info.get("x"), skeleton_info.get("y"), skeleton_info.get("z")]
    else:
        skeleton_position_list = [None, None, None]

    skeleton_position = None if None in skeleton_position_list else np.array(skeleton_position_list)

    if player_position is not None and skeleton_position is not None:
        relative_position = skeleton_position - player_position
        relative_position = np.clip(relative_position, -RELATIVE_DISTANCE_AXIS_MAX, RELATIVE_DISTANCE_AXIS_MAX)
    else:
        relative_position = np.zeros(3)

    standardized_relative_position = relative_position / RELATIVE_DISTANCE_AXIS_MAX

    return standardized_relative_position


def get_player_position(info):
    player_position_list = [info.get("XPos"), info.get("YPos"), info.get("ZPos")]
    player_position = None if None in player_position_list else np.array(player_position_list)
    return player_position


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

        current_index = 0
        self.player_position_start_index = current_index
        player_position = get_player_position(info)
        standardized_position = np.zeros(3) if player_position is None else player_position / ARENA_SIZE
        standardized_position = np.clip(standardized_position, -1, 1)
        current_index += 3

        skeleton_info = get_entity_info(info, ENEMY_TYPE)
        self.skeleton_relative_position_start_index = current_index
        skeleton_relative_position = get_standardized_relative_position(skeleton_info, player_position)
        current_index += 3

        self.direction_vector_start_index = current_index
        direction_vector = get_direction_vector(info)
        current_index += 3

        self.player_life_index = current_index
        player_life = info.get("Life", 0)
        player_life = player_life / PLAYER_MAX_LIFE
        current_index += 1

        self.skeleton_life_index = current_index
        skeleton_life = 0 if skeleton_info is None else skeleton_info.get("life", 0)
        skeleton_life = skeleton_life / ENEMY_MAX_LIFE
        current_index += 1

        self.surroundings_list_index = current_index
        surroundings_list = info.get("Surroundings")
        current_index += 1

        if surroundings_list is not None:
            surroundings = get_grid_obs_vector(surroundings_list)
        else:
            surroundings = np.zeros(GRID_SIZE)

        self.vector = np.hstack((
            standardized_position,
            skeleton_relative_position,
            direction_vector,
            player_life,
            skeleton_life,
            surroundings
        ))

    @staticmethod
    def get_observation_space():
        low_position = -np.ones(3)
        low_relative_position = -np.ones(3)
        low_direction = -np.ones(3)
        low_player_life = 0
        low_skeleton_life = 0
        low_surroundings = np.zeros(GRID_SIZE)
        low = np.hstack((
            low_position,
            low_relative_position,
            low_direction,
            low_player_life,
            low_skeleton_life,
            low_surroundings
        ))

        high_position = np.ones(3)
        high_relative_position = np.ones(3)
        high_direction = np.ones(3)
        high_player_life = 1
        high_skeleton_life = 1
        high_surroundings = len(game_objects) * np.ones(GRID_SIZE)
        high = np.hstack((
            low_relative_position,
            high_relative_position,
            high_direction,
            high_player_life,
            high_skeleton_life,
            high_surroundings
        ))

        return Box(low, high)
