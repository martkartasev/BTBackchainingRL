from py_trees.common import Status
from py_trees.composites import Selector

from bt.actions import Action
from bt.conditions import Condition
from bt.sequence import Sequence


def tree_to_string(tree, depth=0):
    is_running = False
    symbol = "[B]"
    if isinstance(tree, Sequence):
        symbol = "[>]"
    elif isinstance(tree, Selector):
        symbol = "[?]"
    elif isinstance(tree, Condition):
        symbol = "(C)"
    elif isinstance(tree, Action):
        if tree.status != Status.INVALID:
            symbol = "[A]"
            is_running = True
        else:
            symbol = "[a]"
    indent = "    " * (depth - 1)
    if is_running:
        indent = "----" * (depth - 1)
    arrow = "--> " if depth > 0 else ""
    output = f"{indent}{arrow}{symbol} {tree.name} \n"
    for child in tree.children:
        output += tree_to_string(child, depth + 1)
    return output
