import json

import numpy as np
from gym.spaces import Box, Dict, Discrete

CIRCLE_DEGREES = 360

RELATIVE_DISTANCE_AXIS_MAX = 50
INVENTORY_SIZE = 41

PLAYER_MAX_LIFE = 100
PLAYER_MAX_FOOD = 20

ENEMY_TYPE = "Skeleton"
ANIMAL_TYPE = "Cow"
FOOD_TYPES = ["beef", "cooked_beef"]
ENEMY_MAX_LIFE = 24

GRID_SIZE_AXIS = [5, 1, 5]
GRID_SIZE = np.prod(GRID_SIZE_AXIS)

# Always append at the end of this list
game_objects = ["dirt", "grass", "stone", "fire", "air", "brick_block", "netherrack"]


def get_entity_info(info, entity_names):
    if "Entities" in info:
        for entity in info["Entities"]:
            if entity.get("name") in entity_names:
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


def get_surroundings(grid):
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


def get_simplified_surroundings(grid):
    grid_ordinals = [get_simplified_game_object_ordinal(block) for block in grid]
    return np.array(grid_ordinals, dtype=np.float32)


def get_simplified_game_object_ordinal(game_object):
    if game_object is None:
        return 0
    elif game_object == "air":
        return 0
    elif game_object == "fire":
        return 1
    else:
        return 2


def get_standardized_relative_position(entity_info, player_position):
    if entity_info is not None:
        entity_position_list = [entity_info.get("x"), entity_info.get("y"), entity_info.get("z")]
    else:
        entity_position_list = [None, None, None]

    entity_position = None if None in entity_position_list else np.array(entity_position_list)

    if player_position is not None and entity_position is not None:
        relative_position = entity_position - player_position
        relative_position = np.clip(relative_position, -RELATIVE_DISTANCE_AXIS_MAX, RELATIVE_DISTANCE_AXIS_MAX)
    else:
        relative_position = RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)

    standardized_relative_position = relative_position / RELATIVE_DISTANCE_AXIS_MAX

    return standardized_relative_position


def get_player_position(info):
    player_position_list = [info.get("XPos"), info.get("YPos"), info.get("ZPos")]
    player_position = None if None in player_position_list else np.array(player_position_list)
    return player_position


def get_item_inventory_index(info, items):
    if "inventory" in info:
        for inventory_slot in info["inventory"]:
            if inventory_slot.get("type") in items:
                return inventory_slot.get("index", -1) + 1
        return 0
    else:
        return 0


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

        self.dict = {}

        player_position = get_player_position(info)

        food_info = get_entity_info(info, FOOD_TYPES)
        entity_info = get_entity_info(info, [ANIMAL_TYPE]) if food_info is None else food_info
        self.dict["entity_relative_position"] = get_standardized_relative_position(entity_info, player_position)

        self.dict["direction"] = get_direction_vector(info)

        self.dict["health"] = np.array([info.get("Life", 0) / PLAYER_MAX_LIFE])
        self.dict["satiation"] = np.array([info.get("Food", 0) / PLAYER_MAX_FOOD])

        entity_life = 0 if entity_info is None else entity_info.get("life", 0)
        self.dict["entity_health"] = np.array([entity_life / ENEMY_MAX_LIFE])

        self.dict["is_entity_pickable"] = 1 if food_info is not None else 0

        self.dict["has_food"] = get_item_inventory_index(info, FOOD_TYPES) > 0

        surroundings_list = info.get("Surroundings")
        if surroundings_list is not None:
            surroundings = get_simplified_surroundings(surroundings_list)
        else:
            surroundings = np.zeros(GRID_SIZE)
        self.dict["surroundings"] = surroundings

    @staticmethod
    def get_observation_space():
        return Dict(
            spaces={
                "entity_relative_position": Box(-1, 1, (3,)),
                "direction": Box(-1, 1, (3,)),
                "health": Box(0, 1, (1,)),
                "satiation": Box(0, 1, (1,)),
                "entity_health": Box(0, 1, (1,)),
                "is_entity_pickable": Discrete(2),
                "has_food": Discrete(2),
                "surroundings": Box(0, 2, (GRID_SIZE,), dtype=np.uint8)
            }
        )
