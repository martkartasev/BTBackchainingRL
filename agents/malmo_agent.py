import time


class MalmoAgent:
    def __init__(self, agent_host):
        self.agent_host = agent_host
        self.moving = 0
        self.strafing = 0
        self.turning = 0
        self.pitching = 0
        self.jumping = 0
        self.attacking = 0
        self.crouching = 0
        self.using = 0

    def pause(self):
        self.agent_host.sendCommand("move 0")
        self.agent_host.sendCommand("turn 0")
        self.agent_host.sendCommand("strafe 0")
        self.agent_host.sendCommand("jump 0")
        self.agent_host.sendCommand("attack 0")

    def resume(self):
        self.agent_host.sendCommand("move " + str(self.moving))
        self.agent_host.sendCommand("turn " + str(self.turning))
        self.agent_host.sendCommand("strafe " + str(self.strafing))
        self.agent_host.sendCommand("jump " + str(self.jumping))
        self.agent_host.sendCommand("attack " + str(self.attacking))

    def continuous_move(self, val):
        self.agent_host.sendCommand("move " + str(val))
        self.moving = val

    def continuous_turn(self, val):
        self.agent_host.sendCommand("turn " + str(val))
        self.turning = val

    def set_yaw(self, val):
        self.agent_host.sendCommand("setYaw " + str(val))

    def continuous_strafe(self, val):
        self.agent_host.sendCommand("strafe " + str(val))
        self.strafing = val

    def continuous_pitch(self, val):
        self.agent_host.sendCommand("pitch " + str(val))
        self.pitching = val

    def set_pitch(self, val):
        self.agent_host.sendCommand("setPitch " + str(val))

    def continuous_jump(self, toggle):
        self.agent_host.sendCommand("jump " + str(toggle))
        self.jumping = toggle

    def attack(self, toggle):
        self.agent_host.sendCommand("attack " + str(toggle))
        self.attacking = toggle

    def crouch(self, toggle):
        self.agent_host.sendCommand("crouch " + str(toggle))
        self.crouching = toggle

    def swap_items(self, position1, position2):
        self.agent_host.sendCommand("swapInventoryItems {0} {1}".format(position1, position2))

    def select_on_hotbar(self, position):
        self.agent_host.sendCommand(f"hotbar.{position + 1} 1")  # press
        self.agent_host.sendCommand(f"hotbar.{position + 1} 0")  # release
        time.sleep(0.1)  # Stupid but necessary

    def use(self, toggle=None):
        if toggle is None:
            self.agent_host.sendCommand("use")
            time.sleep(0.5)  # Stupid but necessary
        elif toggle == 1:
            self.agent_host.sendCommand("use " + str(toggle))
            self.agent_host.sendCommand("use 0")
            time.sleep(0.001)
            self.agent_host.sendCommand("use 0")
            time.sleep(0.5)  # Stupid but necessary
            self.using = toggle

    def craft(self, item):
        self.agent_host.sendCommand("craft " + str(item))
        time.sleep(0.004)

    def move_west(self):
        self.agent_host.sendCommand("movewest 1")

    def move_south(self):
        self.agent_host.sendCommand("movesouth 1")

    def move_east(self):
        self.agent_host.sendCommand("moveeast 1")

    def move_north(self):
        self.agent_host.sendCommand("movenorth 1")