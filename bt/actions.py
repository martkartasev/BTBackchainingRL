import numpy as np
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from mission.minecraft_types import Block
from mission.observation_manager import ObservationDefinition
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
        fire_loc = np.where(self.agent.observation_manager.observation.dict["surroundings"][0:1, 2:5,
                            2:5] == Block.fire.value)  # TODO: Need to calculate based on axis size
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


class DefeatSkeletonManual(Action):
    WEAPON_REACH = 2
    TURN_THRESHOLD = 0.995
    MIN_TURN_SPEED = 0.1
    MAX_TURN_SPEED = 0.7
    MIN_MOVE_SPEED = 0.1
    MAX_MOVE_SPEED = 0.7
    SLOW_DOWN_THRESHOLD = 10

    def __init__(self, agent, name="Defeat Skeleton"):
        super().__init__(name, agent)

    def update(self):
        print("Defeat Skeleton manual")
        distance_vector = self.agent.observation_manager.observation.dict["enemy_relative_position"]

        enemy_relative_position = np.array([distance_vector[0], distance_vector[2]])
        enemy_distance = np.linalg.norm(enemy_relative_position)
        if enemy_distance != 0:
            enemy_relative_position = enemy_relative_position / enemy_distance

        direction_dot = enemy_relative_position[0]

        sign = 1 if enemy_relative_position[1] > 0 else -1
        if direction_dot <= DefeatSkeletonManual.TURN_THRESHOLD:
            turn_speed = DefeatSkeletonManual.MAX_TURN_SPEED * (1 - direction_dot) / 2
            turn_speed = sign * max(turn_speed, DefeatSkeletonManual.MIN_TURN_SPEED)
            self.agent.continuous_move(0)
            self.agent.continuous_turn(turn_speed)
            return Status.RUNNING
        self.agent.continuous_turn(0)

        distance = distance_vector[0] * ObservationDefinition.RELATIVE_DISTANCE_AXIS_MAX
        if distance > DefeatSkeletonManual.WEAPON_REACH:
            if distance > DefeatSkeletonManual.SLOW_DOWN_THRESHOLD:
                move_speed = DefeatSkeletonManual.MAX_MOVE_SPEED
            else:
                move_speed = DefeatSkeletonManual.MAX_MOVE_SPEED * distance / DefeatSkeletonManual.SLOW_DOWN_THRESHOLD
                move_speed = max(move_speed, DefeatSkeletonManual.MIN_MOVE_SPEED)
            self.agent.continuous_move(move_speed)
            return Status.RUNNING
        self.agent.continuous_move(0)

        self.agent.attack(1)
        return Status.RUNNING

    def terminate(self, new_status):
        self.agent.continuous_turn(0)
        self.agent.continuous_move(0)
        self.agent.attack(0)


class Eat(Action):
    FOOD_INDEX = 1

    def __init__(self, agent, name="Eat"):
        super().__init__(name, agent)
        self.is_running = False

    def update(self):
        if self.agent.observation_manager.observation.dict["satiation"] == 1:
            return Status.SUCCESS

        food_inventory_item = self.agent.observation_manager.observation.dict["has_food"]
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
        distance_vector = self.agent.observation_manager.observation.dict["entity_relative_position"]
        entity_direction_vector = np.array([distance_vector[0], distance_vector[2]])
        entity_direction_vector = entity_direction_vector / np.linalg.norm(entity_direction_vector)

        self.agent.move_towards_flat_direction(entity_direction_vector)

        has_food = self.agent.observation_manager.observation.dict["has_food"] > 0
        return Status.SUCCESS if has_food else Status.RUNNING

    def terminate(self, new_status):
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
    def __init__(self, agent, name="Stop Moving"):
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
