import numpy as np
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from mission.minecraft_types import Block
from mission.observation_manager import ObservationDefinition


class Condition(Behaviour):
    def __init__(self, name, agent):
        super(Condition, self).__init__(name)
        self.agent = agent
        self.update_aux = self.execution
        self.evaluation_manager = None

    def update(self):
        return self.update_aux()

    def evaluation(self):
        result = self.execution()
        self.evaluation_manager.record_node(self, result)
        return result

    def set_evaluation_manager(self, evaluation_manager):
        self.update_aux = self.evaluation
        self.evaluation_manager = evaluation_manager
        self.evaluation_manager.register_node(self)

    def execution(self):
        if self.evaluate(self.agent):
            return Status.SUCCESS
        return Status.FAILURE

    def evaluate(self, agent) -> bool:
        raise NotImplementedError()


class IsSafeFromFire(Condition):

    def __init__(self, agent, name=f"Is safe from fire"):
        super(IsSafeFromFire, self).__init__(name, agent)

    def evaluate(self, agent) -> bool:
        fire_loc = np.where(self.agent.observation_manager.observation.dict["surroundings"][0:1, 2:5, 2:5] == Block.fire.value)  # TODO: Need to calculate based on axis size
        count = len(fire_loc[0])
        if count > 0:
            for i in range(0, count):
                delta_pos = np.array([-1 + fire_loc[2][i], -1 + fire_loc[0][i], -1 + fire_loc[1][i]])
                position = self.agent.observation_manager.observation.dict["position"]
                loc = np.floor(position + delta_pos) + np.array([0.5, 0, 0.5])
                if np.max(np.abs(position - loc)) <= self.agent.observation_manager.helper.definition.FIRE_AVOID_DISTANCE:
                    return False
        return True


class IsEnemyDefeated(Condition):
    def __init__(self, agent):
        super(IsEnemyDefeated, self).__init__(f"Is enemy defeated", agent)

    def evaluate(self, agent) -> bool:
        enemy_life = self.agent.observation_manager.observation.dict["enemy_health"]
        return enemy_life == 0


class IsNotAttackedByEnemy(Condition):
    ENEMY_AGGRO_RANGE = 5

    def __init__(self, agent):
        super(IsNotAttackedByEnemy, self).__init__(f"Is not attacked by enemy", agent)

    def evaluate(self, agent) -> bool:
        enemy_distance = self.agent.observation_manager.observation.dict["enemy_relative_position"]

        non_standardized_distance = enemy_distance * ObservationDefinition.RELATIVE_DISTANCE_AXIS_MAX
        distance = np.linalg.norm(non_standardized_distance)

        return distance >= IsNotAttackedByEnemy.ENEMY_AGGRO_RANGE


class IsEntityVisible(Condition):
    def __init__(self, agent):
        super(IsEntityVisible, self).__init__(f"Is Entity Visible", agent)

    def evaluate(self, agent) -> bool:
        return self.agent.observation_manager.observation.dict["entity_visible"] == 1


class IsCloseToEntity(Condition):
    RANGE = 2

    def __init__(self, agent):
        super(IsCloseToEntity, self).__init__(f"Is Close To Entity", agent)

    def evaluate(self, agent) -> bool:
        entity_distance = self.agent.observation_manager.observation.dict["entity_relative_position"]

        non_standardized_distance = entity_distance * ObservationDefinition.RELATIVE_DISTANCE_AXIS_MAX
        distance = np.linalg.norm(non_standardized_distance)

        return distance <= IsCloseToEntity.RANGE


class IsNotHungry(Condition):
    def __init__(self, agent):
        super(IsNotHungry, self).__init__(f"Is not hungry", agent)

    def evaluate(self, agent) -> bool:
        return self.agent.observation_manager.observation.dict["satiation"] == 1


class HasFood(Condition):
    def __init__(self, agent):
        super(HasFood, self).__init__(f"Has food", agent)

    def evaluate(self, agent) -> bool:
        return self.agent.observation_manager.observation.dict["has_food"] > 0


class IsEntityPickable(Condition):
    def __init__(self, agent):
        super(IsEntityPickable, self).__init__(f"Is entity pickable", agent)

    def evaluate(self, agent) -> bool:
        return self.agent.observation_manager.observation.dict["is_entity_pickable"] == 1
