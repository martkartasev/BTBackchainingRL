import time

import numpy as np
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from observation import GRID_SIZE_AXIS, game_objects


class Action(Behaviour):
    def __init__(self, name, agent=None):
        super(Action, self).__init__(name)
        self.agent = agent


class AvoidFire(Action):
    def __init__(self, agent, name="Avoid Fire"):
        super().__init__(name, agent)
        self.position_in_grid = np.array([int(axis / 2) for axis in GRID_SIZE_AXIS])

    def update(self):
        # Find closest not fire
        grid = self.grid_observation_from_list()
        grid = grid[:, 0, :]

        is_air = (grid == game_objects.index("air"))
        positions = np.argwhere(is_air)

        if len(positions) == 0:
            return Status.FAILURE
        distances_vector = positions - np.array([self.position_in_grid[0], self.position_in_grid[2]])
        distances = np.linalg.norm(distances_vector, axis=1)
        min_dist_arg = np.argmin(distances)

        if distances[min_dist_arg] == 0:
            return Status.SUCCESS

        min_distance_vector = distances_vector[min_dist_arg]
        distance_vector_direction = min_distance_vector / np.linalg.norm(min_distance_vector)

        direction_vector_start_index = self.agent.observation.direction_vector_start_index
        direction_vector_end_index = direction_vector_start_index + 3
        direction_vector = self.agent.observation.vector[direction_vector_start_index:direction_vector_end_index]
        flat_direction_vector = np.array([direction_vector[0], direction_vector[2]])
        flat_direction_vector /= np.linalg.norm(flat_direction_vector)

        angle = np.arccos(np.dot(distance_vector_direction, flat_direction_vector))

        self.agent.continuous_move(np.cos(angle))
        self.agent.continuous_strafe(-np.sin(angle))

        return Status.SUCCESS if grid[0, 0] == game_objects.index("air") else Status.FAILURE

    def grid_observation_from_list(self):
        grid_observation_list = self.agent.observation.vector[self.agent.observation.surroundings_list_index:]

        grid = np.array(grid_observation_list).reshape((GRID_SIZE_AXIS[1], GRID_SIZE_AXIS[2], GRID_SIZE_AXIS[0]))
        grid = np.transpose(grid, (2, 0, 1))
        return grid

    def terminate(self, new_status):
        self.agent.continuous_move(0)
        self.agent.continuous_strafe(0)


class Eat(Action):

    def __init__(self, agent, name="Eat"):
        super().__init__(name, agent)

    def update(self):
        food_inventory_item = self.agent.observation.vector[self.agent.observation.food_inventory_index_index]
        temp_inventory_spot = 3
        self.agent.swap_items(0, temp_inventory_spot)
        self.agent.swap_items(0, food_inventory_item)

        self.agent.use()
        eat_time = 1
        time.sleep(eat_time)

        self.agent.swap_items(0, temp_inventory_spot)

        return Status.SUCCESS


class PickUpEntity(Action):

    def __init__(self, agent, name="Pick up Entity"):
        super().__init__(name, agent)

    def update(self):
        distance_start_index = self.agent.observation.entity_relative_position_start_index
        distance_end_index = self.agent.observation.entity_relative_position_start_index + 3

        distance_vector = self.agent.observation.vector[distance_start_index:distance_end_index]
        entity_direction_vector = distance_vector/np.linalg.norm(distance_vector)

        direction_vector_start_index = self.agent.observation.direction_vector_start_index
        direction_vector_end_index = direction_vector_start_index + 3
        direction_vector = self.agent.observation.vector[direction_vector_start_index:direction_vector_end_index]
        flat_direction_vector = np.array([direction_vector[0], direction_vector[2]])
        flat_direction_vector /= np.linalg.norm(flat_direction_vector)

        angle = np.arccos(np.dot(entity_direction_vector, flat_direction_vector))

        self.agent.continuous_move(np.cos(angle))
        self.agent.continuous_strafe(-np.sin(angle))

        return Status.SUCCESS


class MoveForward(Action):
    def __init__(self, agent, name="Move Forward"):
        super().__init__(name, agent)

    def update(self):
        # print("Forward 1")
        self.agent.continuous_move(1)
        return Status.RUNNING

    def terminate(self, new_status):
        #   print("Forward 0")
        self.agent.continuous_move(0)


class MoveBackward(Action):
    def __init__(self, agent, name="Move Backward"):
        super().__init__(name, agent)

    def update(self):
        #   print("Forward -1")
        self.agent.continuous_move(-1)
        return Status.RUNNING

    def terminate(self, new_status):
        # print("Forward 0")
        self.agent.continuous_move(0)


class MoveLeft(Action):
    def __init__(self, agent, name="Move Left"):
        super().__init__(name, agent)

    def update(self):
        #  print("Strafe -1")
        self.agent.continuous_strafe(-1)
        return Status.RUNNING

    def terminate(self, new_status):
        #   print("Strafe 0")
        self.agent.continuous_strafe(0)


class MoveRight(Action):
    def __init__(self, agent, name="Move Right"):
        super().__init__(name, agent)

    def update(self):
        #  print("Strafe 1")
        self.agent.continuous_strafe(1)
        return Status.RUNNING

    def terminate(self, new_status):
        #   print("Strafe 0")
        self.agent.continuous_strafe(0)


class TurnLeft(Action):
    def __init__(self, agent, name="Turn Left"):
        super().__init__(name, agent)

    def update(self):
        #   print("Turn -1")
        self.agent.continuous_turn(-1)
        return Status.RUNNING

    def terminate(self, new_status):
        #  print("Turn 0")
        self.agent.continuous_turn(0)


class TurnRight(Action):
    def __init__(self, agent, name="Turn Right"):
        super().__init__(name, agent)

    def update(self):
        # print("Turn 1")
        self.agent.continuous_turn(1)
        return Status.RUNNING

    def terminate(self, new_status):
        # print("Turn 0")
        self.agent.continuous_turn(0)


class PitchUp(Action):
    def __init__(self, agent, name="Pitch Up"):
        super().__init__(name, agent)

    def update(self):
        self.agent.continuous_pitch(-1)
        return Status.RUNNING

    def terminate(self, new_status):
        self.agent.continuous_pitch(0)


class PitchDown(Action):
    def __init__(self, agent, name="Pitch Down"):
        super().__init__(name, agent)

    def update(self):
        self.agent.continuous_pitch(1)
        return Status.RUNNING

    def terminate(self, new_status):
        self.agent.continuous_pitch(0)


class Jump(Action):
    def __init__(self, agent, name="Jump"):
        super().__init__(name, agent)

    def update(self):
        self.agent.continuous_jump(1)
        return Status.RUNNING

    def terminate(self, new_status):
        self.agent.continuous_jump(0)


class Attack(Action):
    def __init__(self, agent, name="Attack"):
        super().__init__(name, agent)

    def update(self):
        #  print("Attack 1")
        self.agent.attack(1)
        return Status.RUNNING

    def terminate(self, new_status):
        #  print("Attack 0")
        self.agent.attack(0)


class Crouch(Action):
    def __init__(self, agent, name="Crouch"):
        super().__init__(name, agent)

    def update(self):
        self.agent.crouch(1)
        return Status.RUNNING

    def terminate(self, new_status):
        self.agent.crouch(0)


class StopMoving(Action):
    def __init__(self, agent, name="Move Left"):
        super().__init__(name, agent)

    def update(self):
        self.agent.continuous_move(0)
        self.agent.continuous_strafe(0)
        self.agent.continuous_turn(0)
        self.agent.continuous_pitch(0)
        return Status.RUNNING

    def terminate(self, new_status):
        #   print("Strafe 0")
        self.agent.continuous_strafe(0)


class Use(Action):
    def __init__(self, agent, name="Use"):
        super().__init__(name, agent)

    def update(self):
        self.agent.use()
        return Status.RUNNING
