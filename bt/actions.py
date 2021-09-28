import numpy as np
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from minecraft_types import Block
from observation import GRID_SIZE_AXIS, game_objects
from utils.linalg import rotation_matrix_y


class Action(Behaviour):
    def __init__(self, name, agent=None):
        super(Action, self).__init__(name)
        self.agent = agent


class AvoidFire(Action):
    def __init__(self, agent, name="AvoidFire"):
        super().__init__(name, agent)
        self.result = np.array([0.0, 0.0, 0.0])

    def update(self):
        fire_loc = np.where(self.agent.observation_manager.observation.dict["surroundings"][0:1, 2:5, 2:5] == Block.fire.value)  # TODO: Need to calculate based on axis size
        count = len(fire_loc[0])
        if count > 0:
            d_res = np.array([0.0, 0.0, 0.0])

            for i in range(0, count):
                pos = self.agent.observation_manager.observation.dict["position"]

                delta_pos = np.array([-1 + fire_loc[2][i], -1 + fire_loc[0][i], -1 + fire_loc[1][i]])
                loc = np.floor(pos + delta_pos) + np.array([0.5, 0, 0.5])
                d_res += (pos - loc)

            self.result = rotation_matrix_y(self.agent.observation_manager.observation.dict["euler_direction"][0]).dot(d_res) * 2

        self.agent.continuous_move(self.result[2])
        self.agent.continuous_strafe(-self.result[0])

        return Status.RUNNING

    def terminate(self, new_status):
        self.agent.continuous_move(0)
        self.agent.continuous_strafe(0)


class Eat(Action):
    FOOD_INDEX = 1

    def __init__(self, agent, name="Eat"):
        super().__init__(name, agent)
        self.is_running = False

    def update(self):
        if self.agent.observation.dict["satiation"] == 1:
            return Status.SUCCESS

        food_inventory_item = self.agent.observation.dict["has_food"]
        if food_inventory_item == 0:
            return Status.FAILURE

        if not self.is_running:
            self.agent.select_on_hotbar(Eat.FOOD_INDEX)
            self.is_running = True

        self.agent.continuous_use(1)
        return Status.RUNNING

    def terminate(self, new_status):
        self.is_running = False
        self.agent.continuous_use(0)
        self.agent.select_on_hotbar(0)


class PickUpEntity(Action):

    def __init__(self, agent, name="Pick up Entity"):
        super().__init__(name, agent)

    def update(self):
        distance_vector = self.agent.observation.dict["entity_relative_position"]
        entity_direction_vector = np.array([distance_vector[0], distance_vector[2]])
        entity_direction_vector = entity_direction_vector / np.linalg.norm(entity_direction_vector)

        self.agent.move_towards_flat_direction(entity_direction_vector)

        has_food = self.agent.observation.dict["has_food"] > 0
        return Status.SUCCESS if has_food else Status.RUNNING

    def terminate(self, new_status):
        #   print("Forward 0")
        self.agent.continuous_move(0)
        self.agent.continuous_strafe(0)


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


class Use(Action):
    def __init__(self, agent, name="Use"):
        super().__init__(name, agent)

    def update(self):
        self.agent.use()
        return Status.RUNNING


# TODO: Remove
class ActionPlaceholder(Action):
    def __init__(self, agent, name="Placeholder"):
        super().__init__(name, agent)

    def update(self):
        return Status.SUCCESS
