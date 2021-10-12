import json

import numpy as np
from gym.spaces import Box, Dict, Discrete

from minecraft_types import Block, Enemy

CIRCLE_DEGREES = 360

ARENA_SIZE = 16
RELATIVE_DISTANCE_AXIS_MAX = 20
INVENTORY_SIZE = 41

PLAYER_MAX_LIFE = 20
PLAYER_MAX_FOOD = 20

ENEMY_TYPE = "Skeleton"
ANIMAL_TYPE = "Cow"
FOOD_TYPES = ["beef", "cooked_beef"]
ENEMY_MAX_LIFE = 24

GRID_SIZE_AXIS = [1, 7, 7]
GRID_SIZE = np.prod(GRID_SIZE_AXIS)

# Always append at the end of this list
game_objects = ["dirt", "grass", "stone", "fire", "air", "brick_block", "netherrack"]  # TODO: I would prefer an enum. I brought in minecraft_types but its a little WIP


def get_entity_info(info, entity_names):
    if "Entities" in info:
        for entity in info["Entities"]:
            if entity.get("name") in entity_names:
                return entity


def get_yaw(info):
    if "Yaw" in info:
        return bound_degrees(info["Yaw"])
    return 0


def bound_degrees(yaw):
    if 180 < yaw < -180:
        yaw = yaw % CIRCLE_DEGREES

    if yaw < 0:
        yaw = CIRCLE_DEGREES + yaw

    return yaw


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


def get_euler_direction(info):
    yaw = get_yaw(info)
    pitch = get_pitch(info)

    if yaw is None or pitch is None:
        return np.zeros(3)

    return [yaw, pitch, 0]  # yaw, pitch, roll


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
    grid_ordinals = [Block.get_simplified_game_object_ordinal(block) for block in grid]
    return np.array(grid_ordinals, dtype=np.float32)


def get_relative_position(entity_info, player_position):
    if player_position is None:
        return RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)
    if entity_info is None:
        return RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)

    entity_position_list = get_entity_position(entity_info)
    if None in entity_position_list:
        return RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)

    entity_position = np.array(entity_position_list)
    relative_position = entity_position - player_position
    return np.clip(relative_position, -RELATIVE_DISTANCE_AXIS_MAX, RELATIVE_DISTANCE_AXIS_MAX)


def get_entity_position(entity_info):
    if entity_info is not None:
        return [entity_info.get("x"), entity_info.get("y"), entity_info.get("z")]
    return [0, 0, 0]


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
    player_position = [0, 0, 0] if None in player_position_list else np.array(player_position_list)
    return player_position


def get_item_inventory_index(info, items):
    if "inventory" in info:
        for inventory_slot in info["inventory"]:
            if inventory_slot.get("type") in items:
                return inventory_slot.get("index", -1) + 1
        return 0
    else:
        return 0


def get_y_rotation_from(a, b):
    if a is not None and b is not None:
        dx = b[0] - a[0]
        dz = b[2] - a[2]
        yaw = -180 * np.arctan2(dx, dz) / np.pi
        return bound_degrees(yaw)
    return 0


class ObservationManager:

    def __init__(self, observation_filter=None):
        self.previous_observation = None
        self.observation = None
        self.reward = 0
        self.index = 0

        self.observation_filter = observation_filter

    def update(self, observations, reward):
        self.previous_observation = self.observation
        self.observation = Observation(observations, self.observation_filter)

        self.reward = reward

        self.index += 1
        return self.observation

    def get_observation_space(self):
        return Observation.get_observation_space(self.observation_filter)

    def reset(self):
        self.previous_observation = None
        self.observation = None
        self.reward = 0
        self.index = 0

    def is_agent_alive(self):
        return self.observation.dict["health"] > 0


class Observation:

    def __init__(self, observations, observation_filter=None):
        if observations is None or len(observations) == 0:
            print("Observations is null or empty")
            return

        info_json = observations[-1].text
        if info_json is None:
            print("Info is null")
            return

        info = json.loads(info_json)

        observation_dict = {}

        position = get_player_position(info)
        direction = get_direction_vector(info)
        euler_direction = get_euler_direction(info)
        observation_dict["position"] = position
        observation_dict["direction"] = direction
        observation_dict["euler_direction"] = euler_direction

        enemy_info = get_entity_info(info, [ENEMY_TYPE])
        rotated_position = get_standardized_rotated_position(enemy_info, position, direction)
        observation_dict["enemy_relative_position"] = rotated_position

        yaw = get_yaw(info)
        delta = get_relative_position(enemy_info, position)
        rot = np.radians(get_y_rotation_from(position, get_entity_position(enemy_info)) - yaw)
        observation_dict["enemy_relative_distance"] = np.array(np.linalg.norm(np.array(delta[0], delta[2])) / RELATIVE_DISTANCE_AXIS_MAX)
        observation_dict["enemy_relative_direction"] = np.array([((np.cos(rot) + 1) / 2), ((np.sin(rot) + 1) / 2), ])
        observation_dict["enemy_targeted"] = 'LineOfSight' in info.keys() and Enemy.is_enemy(info.get("LineOfSight").get("type"))

        food_info = get_entity_info(info, FOOD_TYPES)
        entity_info = get_entity_info(info, [ANIMAL_TYPE]) if food_info is None else food_info
        observation_dict["entity_relative_position"] = get_standardized_rotated_position(entity_info, position, direction)

        observation_dict["health"] = np.array([info.get("Life", 0) / PLAYER_MAX_LIFE])
        observation_dict["satiation"] = np.array([info.get("Food", 0) / PLAYER_MAX_FOOD])

        enemy_health = enemy_info.get("life", 0) / ENEMY_MAX_LIFE if enemy_info is not None else 0
        observation_dict["enemy_health"] = np.array([enemy_health])

        entity_visible = 1 if entity_info is not None else 0
        observation_dict["entity_visible"] = np.array([entity_visible])

        observation_dict["is_entity_pickable"] = 1 if food_info is not None else 0
        observation_dict["has_food"] = get_item_inventory_index(info, FOOD_TYPES) > 0

        surroundings_list = info.get("Surroundings")
        if surroundings_list is not None:
            surroundings = get_simplified_surroundings(surroundings_list).reshape(GRID_SIZE_AXIS)
        else:
            surroundings = np.zeros(GRID_SIZE).reshape(GRID_SIZE_AXIS)
        observation_dict["surroundings"] = surroundings

        self.dict = observation_dict
        if observation_filter is not None:
            self.filtered = {key: value for key, value in observation_dict.items() if key in observation_filter}
            surroundings = self.filtered["surroundings"]
            self.filtered["surroundings"] = np.rot90(surroundings, int((euler_direction[0] + 45) / 90) + 2, axes=(1, 2)).ravel()

    @staticmethod
    def get_observation_space(observation_filter=None):
        full_space = {
            "entity_relative_position": Box(-1, 1, (3,)),
            "enemy_relative_position": Box(-1, 1, (3,)),

            "enemy_relative_distance": Box(0, 1, (1,)),
            "enemy_targeted": Box(0, 1, (1,), dtype=np.uint8),
            "enemy_relative_direction": Box(0, 1, (2,)),

            "direction": Box(0, 1, (3,)),
            "health": Box(0, 1, (1,)),
            "satiation": Box(0, 1, (1,)),
            "enemy_health": Box(0, 1, (1,)),
            "entity_visible": Box(0, 1, (1,), dtype=np.uint8),
            "is_entity_pickable": Discrete(2),
            "has_food": Discrete(2),
            "surroundings": Box(0, 2, (GRID_SIZE,), dtype=np.uint8)
        }
        if observation_filter is None:
            return Dict(spaces=full_space)
        else:
            reduced_space = {key: value for key, value in full_space.items() if key in observation_filter}
            return Dict(spaces=reduced_space)

#  Temp reference from my old repo


#     def get_observation_array(self):
#         if self.agent is not None and self.agent.observations is not None and self.agent.observations.lookToPosition is not None:
#             delta = self.agent.observations.position - self.agent.observations.lookToPosition
#             observations_yaw = self.agent.observations.yaw
#             rot = np.radians(get_y_rotation_from(self.agent, self.agent.observations.lookToPosition) - observations_yaw)
#             yaw = np.radians(observations_yaw)
#             return {
#                 "cont": np.array([
#                     self.agent.observations.agentEntity.life / 20,
#                     self.agent.observations.enemyEntity.life / 20,
#                     np.linalg.norm(np.array(delta[0], delta[2])) / 60,
#                     ((np.cos(rot) + 1) / 2),
#                     ((np.sin(rot) + 1) / 2),
#                     (self.agent.observations.pitch + 90) / 180,
#                     self.agent.observations.LineOfSight is not None and Enemy.is_enemy(self.agent.observations.LineOfSight.type)]),
#                 "disc": np.rot90(self.agent.observations.near, int((observations_yaw + 45) / 90) + 2, axes=(1, 2)).ravel()}
#         else:
#             return {"cont": np.array([0, 0, 0, 0, 0, 0, 0]),
#                     "disc": np.full((1, 7, 7), 0).ravel()}
