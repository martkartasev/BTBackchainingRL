import json
from dataclasses import dataclass

import numpy as np
from gym.spaces import Box, Dict, Discrete

from minecraft_types import Block, Enemy


@dataclass
class ObservationDefinition:  # Override defaults from main spec
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

    GRID_SIZE_AXIS = [1, 11, 11]
    GRID_SIZE = np.prod(GRID_SIZE_AXIS)

    # Always append at the end of this list
    GAME_OBJECTS = ["dirt", "grass", "stone", "fire", "air", "brick_block", "netherrack"]  # TODO: I would prefer an enum. I brought in minecraft_types but its a little WIP


@dataclass
class RewardDefinition:  # Override defaults from main spec
    POST_CONDITION_FULFILLED_REWARD: int = 1000
    AGENT_DEAD_REWARD: int = -1000
    ACC_VIOLATED_REWARD: int = -1000
    STEP_REWARD: int = -0.1


class ObservationManager:

    def __init__(self, observation_filter=None, reward_definition=RewardDefinition(), observation_definition=ObservationDefinition()):
        self.previous_observation = None
        self.observation = None
        self.reward = 0
        self.index = 0
        self.reward_definition = reward_definition
        self.observation_filter = observation_filter
        self.helper = ObservationHelper(observation_definition)

    def update(self, observations, reward):
        self.previous_observation = self.observation
        self.observation = Observation(observations, self.helper, self.observation_filter)

        self.reward = reward

        self.index += 1
        return self.observation

    def get_observation_space(self):
        return Observation.get_observation_space(self.helper.definition, self.observation_filter)

    def reset(self):
        self.previous_observation = None
        self.observation = None
        self.reward = 0
        self.index = 0

    def is_agent_alive(self):
        return self.observation.dict["health"] > 0


class Observation:

    def __init__(self, observations, helper, observation_filter=None):
        self.definition = helper.definition

        if observations is None or len(observations) == 0:
            print("Observations is null or empty")
            return

        info_json = observations[-1].text
        if info_json is None:
            print("Info is null")
            return

        info = json.loads(info_json)

        observation_dict = {}

        position = helper.get_player_position(info)
        direction = helper.get_direction_vector(info)
        euler_direction = helper.get_euler_direction(info)
        observation_dict["position"] = position
        observation_dict["direction"] = direction
        observation_dict["euler_direction"] = euler_direction

        enemy_info = helper.get_entity_info(info, [helper.definition.ENEMY_TYPE])
        rotated_position = helper.get_standardized_rotated_position(enemy_info, position, direction)
        observation_dict["enemy_relative_position"] = rotated_position

        yaw = helper.get_yaw(info)
        delta = helper.get_relative_position(enemy_info, position)
        rot = np.radians(helper.get_y_rotation_from(position, helper.get_entity_position(enemy_info)) - yaw)
        observation_dict["enemy_relative_distance"] = np.array([np.linalg.norm(np.array(delta[0], delta[2])) / self.definition.RELATIVE_DISTANCE_AXIS_MAX])
        observation_dict["enemy_relative_direction"] = np.array([((np.cos(rot) + 1) / 2), ((np.sin(rot) + 1) / 2), ])
        observation_dict["enemy_targeted"] = np.array(['LineOfSight' in info.keys() and Enemy.is_enemy(info.get("LineOfSight").get("type"))])

        food_info = helper.get_entity_info(info, self.definition.FOOD_TYPES)
        entity_info = helper.get_entity_info(info, [self.definition.ANIMAL_TYPE]) if food_info is None else food_info
        observation_dict["entity_relative_position"] = helper.get_standardized_rotated_position(entity_info, position, direction)

        observation_dict["health"] = np.array([info.get("Life", 0) / self.definition.PLAYER_MAX_LIFE])
        observation_dict["satiation"] = np.array([info.get("Food", 0) / self.definition.PLAYER_MAX_FOOD])

        enemy_health = enemy_info.get("life", 0) / self.definition.ENEMY_MAX_LIFE if enemy_info is not None else 0
        observation_dict["enemy_health"] = np.array([enemy_health])

        entity_visible = 1 if entity_info is not None else 0
        observation_dict["entity_visible"] = np.array([entity_visible])

        observation_dict["is_entity_pickable"] = 1 if food_info is not None else 0
        observation_dict["has_food"] = helper.get_item_inventory_index(info, self.definition.FOOD_TYPES) > 0

        surroundings_list = info.get("Surroundings")
        if surroundings_list is not None:
            surroundings = helper.get_simplified_surroundings(surroundings_list).reshape(self.definition.GRID_SIZE_AXIS)
        else:
            surroundings = np.zeros(self.definition.GRID_SIZE).reshape(self.definition.GRID_SIZE_AXIS)
        observation_dict["surroundings"] = surroundings

        self.dict = observation_dict
        if observation_filter is not None:
            self.filtered = {key: value for key, value in observation_dict.items() if key in observation_filter}
            surroundings = self.filtered["surroundings"]
            self.filtered["surroundings"] = np.rot90(surroundings, int((euler_direction[0] + 45) / 90) + 2, axes=(1, 2)).ravel()

    @staticmethod
    def get_observation_space(observation_definition, observation_filter=None):
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
            "surroundings": Box(0, 2, (observation_definition.GRID_SIZE,), dtype=np.uint8)
        }
        if observation_filter is None:
            return Dict(spaces=full_space)
        else:
            reduced_space = {key: value for key, value in full_space.items() if key in observation_filter}
            return Dict(spaces=reduced_space)


class ObservationHelper:
    def __init__(self, observation_definition):
        self.definition = observation_definition

    @staticmethod
    def get_entity_info(info, entity_names):
        if "Entities" in info:
            for entity in info["Entities"]:
                if entity.get("name") in entity_names:
                    return entity

    def get_yaw(self, info):
        if "Yaw" in info:
            return self.bound_degrees(info["Yaw"])
        return 0

    def bound_degrees(self, yaw):
        if 180 < yaw < -180:
            yaw = yaw % self.definition.CIRCLE_DEGREES

        if yaw < 0:
            yaw = self.definition.CIRCLE_DEGREES + yaw

        return yaw

    @staticmethod
    def get_pitch(info):
        return info.get("Pitch")

    def get_direction_vector(self, info):
        yaw = self.get_yaw(info)
        pitch = self.get_pitch(info)

        if yaw is None or pitch is None:
            return np.zeros(3)

        yaw_radians = self.degrees_to_radians(yaw)
        pitch_radians = self.degrees_to_radians(pitch)

        direction_vector = np.array([
            -np.sin(yaw_radians) * np.cos(pitch_radians),
            -np.sin(pitch_radians),
            np.cos(yaw_radians) * np.cos(pitch_radians),
        ])
        return direction_vector

    def get_euler_direction(self, info):
        yaw = self.get_yaw(info)
        pitch = self.get_pitch(info)

        if yaw is None or pitch is None:
            return np.zeros(3)

        return [yaw, pitch, 0]  # yaw, pitch, roll

    def degrees_to_radians(self, deg):
        half_circle = self.definition.CIRCLE_DEGREES / 2
        return deg * np.pi / half_circle

    def get_surroundings(self, grid):
        grid_ordinals = [self.get_game_object_ordinal(block) for block in grid]
        return np.array(grid_ordinals, dtype=np.float32)

    def get_game_object_ordinal(self, game_object):
        if game_object is None:
            return 0
        if game_object not in self.definition.GAME_OBJECTS:
            print(f"Object {game_object} has not been added to the game objects list")
            return 0
        else:
            return self.definition.GAME_OBJECTS.index(game_object) + 1

    @staticmethod
    def get_simplified_surroundings(grid):
        grid_ordinals = [Block.get_simplified_game_object_ordinal(block) for block in grid]
        return np.array(grid_ordinals, dtype=np.float32)

    def get_relative_position(self, entity_info, player_position):
        if player_position is None:
            return self.definition.RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)
        if entity_info is None:
            return self.definition.RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)

        entity_position_list = self.get_entity_position(entity_info)
        if None in entity_position_list:
            return self.definition.RELATIVE_DISTANCE_AXIS_MAX * np.ones(3)

        entity_position = np.array(entity_position_list)
        relative_position = entity_position - player_position
        return np.clip(relative_position, -self.definition.RELATIVE_DISTANCE_AXIS_MAX, self.definition.RELATIVE_DISTANCE_AXIS_MAX)

    def get_entity_position(self, entity_info):
        if entity_info is not None:
            return [entity_info.get("x"), entity_info.get("y"), entity_info.get("z")]
        return [0, 0, 0]

    def get_standardized_rotated_position(self, entity_info, player_position, direction):
        relative_position = self.get_relative_position(entity_info, player_position)

        flat_direction_vector = np.delete(direction, 1)
        flat_direction_vector /= np.linalg.norm(flat_direction_vector)
        side_direction_vector = np.array([flat_direction_vector[1], -flat_direction_vector[0]])

        flat_relative_position = np.delete(relative_position, 1)
        relative_position_length = np.linalg.norm(flat_relative_position)
        relative_position_normalized = flat_relative_position / relative_position_length

        angle = np.arccos(np.dot(relative_position_normalized, flat_direction_vector))
        sign = 1 if np.dot(relative_position_normalized, side_direction_vector) > 0 else -1
        rotated_position = relative_position_length * np.array([np.cos(angle), 0, -sign * np.sin(angle)])

        return np.clip(rotated_position / self.definition.RELATIVE_DISTANCE_AXIS_MAX, -1, 1)

    @staticmethod
    def get_player_position(info):
        player_position_list = [info.get("XPos"), info.get("YPos"), info.get("ZPos")]
        player_position = [0, 0, 0] if None in player_position_list else np.array(player_position_list)
        return player_position

    @staticmethod
    def get_item_inventory_index(info, items):
        if "inventory" in info:
            for inventory_slot in info["inventory"]:
                if inventory_slot.get("type") in items:
                    return inventory_slot.get("index", -1) + 1
            return 0
        else:
            return 0

    def get_y_rotation_from(self, a, b):
        if a is not None and b is not None:
            dx = b[0] - a[0]
            dz = b[2] - a[2]
            yaw = -180 * np.arctan2(dx, dz) / np.pi
            return self.bound_degrees(yaw)
        return 0
