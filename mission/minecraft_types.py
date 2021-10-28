from enum import Enum, IntEnum


class Enemy(Enum):
    Skeleton = 1
    Creeper = 2
    Zombie = 3
    VindicationIllager = 4

    @staticmethod
    def is_enemy(name):
        return any(enemy.name == name for enemy in Enemy)

    @staticmethod
    def get_type(text):
        for enemy in Enemy:
            if enemy.name == text:
                return enemy.value
        raise TypeError()


class Block(IntEnum):
    air = 0
    dirt = 1
    fire = 2
    grass = 3
    log = 4
    planks = 5
    wooden_door = 6
    netherrack = 7
    stone = 8
    bedrock = 9
    brick_block = 10
    diamond_block = 11

    @staticmethod
    def get_simplified_game_object_ordinal(string):
        val = Block[string]
        if val == Block.air:
            return val.value
        if val == Block.fire:
            return val.value
        return Block.dirt.value


class Item(Enum):
    diamond_pickaxe = 0
    diamond_axe = 1
    diamond_shovel = 2
    diamond_hoe = 3
    water_bucket = 4


passable_blocks = ["air",
                   "tallgrass",
                   "double_plant",
                   "red_flower",
                   "yellow_flower",
                   "brown_mushroom"]
