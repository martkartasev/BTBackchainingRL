from py_trees.behaviour import Behaviour
from py_trees.common import Status


class Action(Behaviour):
    def __init__(self, name, agent):
        super(Action, self).__init__(name)
        self.agent = agent


class AvoidFire(Action):
    def __init__(self, agent, name="Avoid Fire"):
        super().__init__(name, agent)

    def update(self):
        # TODO: Implement this
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


class Use(Action):
    def __init__(self, agent, name="Use"):
        super().__init__(name, agent)

    def update(self):
        self.agent.use()
        return Status.RUNNING

