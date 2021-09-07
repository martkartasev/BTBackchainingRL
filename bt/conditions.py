import numpy as np
from py_trees.behaviour import Behaviour
from py_trees.common import Status

import observation
from observation import game_objects, GRID_SIZE_AXIS


class Condition(Behaviour):
    def __init__(self, name, agent=None):
        super(Condition, self).__init__(name)
        self.agent = agent


class IsNotInFire(Condition):
    def __init__(self, agent):
        super(IsNotInFire, self).__init__(f"Is not in fire", agent)

    def update(self):
        grid_list = self.agent.observation.dict["surroundings"]
        me_position_index = int((GRID_SIZE_AXIS[0] * GRID_SIZE_AXIS[2] - 1) / 2)
        return Status.SUCCESS if grid_list[me_position_index] != (game_objects.index("fire") + 1) else Status.FAILURE


class IsEnemyDefeated(Condition):
    def __init__(self, agent):
        super(IsEnemyDefeated, self).__init__(f"Is enemy defeated", agent)

    def update(self):
        enemy_life = self.agent.observation.dict["entity_health"]
        return Status.SUCCESS if enemy_life == 0 else Status.FAILURE


class IsNotAttackedByEnemy(Condition):
    def __init__(self, agent):
        super(IsNotAttackedByEnemy, self).__init__(f"Is not attacked by enemy", agent)

    def update(self):
        enemy_distance = self.agent.observation.dict["entity_relative_position"]

        non_standardized_distance = enemy_distance * observation.RELATIVE_DISTANCE_AXIS_MAX
        distance = np.linalg.norm(non_standardized_distance)

        return Status.SUCCESS if distance >= 2 else Status.FAILURE


class IsNotHungry(Condition):
    def __init__(self, agent):
        super(IsNotHungry, self).__init__(f"Is not hungry", agent)

    def update(self):
        return Status.SUCCESS if self.agent.observation.dict["satiation"] == 1 else Status.FAILURE


class HasFood(Condition):
    def __init__(self, agent):
        super(HasFood, self).__init__(f"Has food", agent)

    def update(self):
        return Status.SUCCESS if self.agent.observation.dict["food_inventory_index"] > 0 else Status.FAILURE


class IsEntityPickable(Condition):
    def __init__(self, agent):
        super(IsEntityPickable, self).__init__(f"Is entity pickable", agent)

    def update(self):
        return Status.SUCCESS if self.agent.observation.dict["is_entity_pickable"] == 1 else Status.FAILURE
