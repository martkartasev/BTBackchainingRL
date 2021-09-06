from py_trees.behaviour import Behaviour
from py_trees.common import Status

from observation import game_objects, GRID_SIZE_AXIS


class Condition(Behaviour):
    def __init__(self, name, agent=None):
        super(Condition, self).__init__(name)
        self.agent = agent


class IsNotInFire(Condition):
    def __init__(self, agent):
        super(IsNotInFire, self).__init__(f"Is not in fire", agent)

    def update(self):
        grid_list = self.agent.observation.vector[self.agent.observation.surroundings_list_index:]
        me_position_index = GRID_SIZE_AXIS[1] * GRID_SIZE_AXIS[2] + 1
        return Status.SUCCESS if grid_list[me_position_index] != (game_objects.index("fire") + 1) else Status.FAILURE


class IsEnemyDefeated(Condition):
    def __init__(self, agent):
        super(IsEnemyDefeated, self).__init__(f"Is enemy defeated", agent)

    def update(self):
        enemy_life = self.agent.observation.vector[self.agent.observation.entity_life_index]
        return Status.SUCCESS if enemy_life == 0 else Status.FAILURE


class IsNotHungry(Condition):
    def __init__(self, agent):
        super(IsNotHungry, self).__init__(f"Is not hungry", agent)

    def update(self):
        player_food = self.agent.observation.vector[self.agent.observation.player_food_index]
        return Status.SUCCESS if player_food == 1 else Status.FAILURE


class HasFood(Condition):
    def __init__(self, agent):
        super(HasFood, self).__init__(f"Has food", agent)

    def update(self):
        has_food =  self.agent.observation.vector[self.agent.observation.food_inventory_index_index] > 0
        return Status.SUCCESS if has_food else Status.FAILURE


class IsEntityPickable(Condition):
    def __init__(self, agent):
        super(IsEntityPickable, self).__init__(f"Is entity pickable", agent)

    def update(self):
        is_entity_pickable = self.agent.observation.vector[self.agent.observation.is_entity_pickable_index]
        is_entity_pickable = (is_entity_pickable == 1)

        return Status.SUCCESS if is_entity_pickable else Status.FAILURE
