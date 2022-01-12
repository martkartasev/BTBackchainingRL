from random import randint


def get_random_in_range(value_range):
    if len(value_range) == 1:
        return value_range[0]
    elif len(value_range) == 2:
        return randint(value_range[0], value_range[1])
    else:
        raise ValueError("value_range must be a list of size 1 or 2.")
