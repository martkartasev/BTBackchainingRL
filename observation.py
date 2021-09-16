import json

import numpy as np
from gym.spaces import Box, Dict

CIRCLE_DEGREES = 360

ARENA_SIZE = 16
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


def get_relative_position(entity_info, player_position):
    if player_position is None:
        return RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)
    if entity_info is None:
        return RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)

    entity_position_list = [entity_info.get("x"), entity_info.get("y"), entity_info.get("z")]
    if None in entity_position_list:
        return RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)

    entity_position = np.array(entity_position_list)
    relative_position = entity_position - player_position
    return np.clip(relative_position, -RELATIVE_DISTANCE_AXIS_MAX, RELATIVE_DISTANCE_AXIS_MAX)


def get_standardized_rotated_position(entity_info, player_position, direction):
    relative_position = get_relative_position(entity_info, player_position)

    flat_direction_vector = np.delete(direction, 1)
    flat_direction_vector /= np.linalg.norm(flat_direction_vector)
    side_direction_vector = np.array([flat_direction_vector[1], -flat_direction_vector[0]])

    flat_relative_position = np.delete(relative_position, 1)
    relative_position_length = np.linalg.norm(flat_relative_position)
    relative_position_normalized = flat_relative_position / relative_position_length

    angle = np.arccos(np.dot(relative_position_normalized, flat_direction_vector))
    sign = 1 if np.dot(relative_position_normalized, side_direction_vector) > 0 else -1
    rotated_position = relative_position_length * np.array([np.cos(angle), 0, -sign * np.sin(angle)])
    
    return np.clip(rotated_position / RELATIVE_DISTANCE_AXIS_MAX, -1, 1)


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

        position = get_player_position(info)
        direction = get_direction_vector(info)
        self.dict["direction"] = direction

        enemy_info = get_entity_info(info, [ENEMY_TYPE])
        self.dict["enemy_relative_position"] = get_standardized_rotated_position(enemy_info, position, direction)

        entity_info = get_entity_info(info, [ANIMAL_TYPE])
        self.dict["entity_relative_position"] = get_standardized_rotated_position(entity_info, position, direction)

        self.dict["health"] = np.array([info.get("Life", 0) / PLAYER_MAX_LIFE])
        enemy_health = enemy_info.get("life", 0) / ENEMY_MAX_LIFE if enemy_info is not None else 0
        self.dict["enemy_health"] = np.array([enemy_health])

        entity_visible = 1 if entity_info is not None else 0
        self.dict["entity_visible"] = np.array([entity_visible])

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
                "enemy_relative_position": Box(-1, 1, (3,)),
                "direction": Box(-1, 1, (3,)),
                "health": Box(0, 1, (1,)),
                "enemy_health": Box(0, 1, (1,)),
                "entity_visible": Box(0, 1, (1,), dtype=np.uint8),
                "surroundings": Box(0, 2, (GRID_SIZE,), dtype=np.uint8)
            }
        )
