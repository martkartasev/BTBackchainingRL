from bt.conditions import Condition
from bt.sequence import Sequence
from py_trees.composites import Selector

from learning.baseline_node import BaselinesNode
from utils.file import create_file_and_write


# Separate implementation from the XML one
def tree_to_drawio_csv(tree):
    graph_attributes = {
        "label": "%step%",
        "style": "shape=%shape%;fontSize=%fontsize%;spacingBottom=%spacingbottom%;",
        "namespace": "csvimport-",
        "connect": '{"from":"refs", "to":"id", "invert":true, "style":"curved=0;endArrow=blockThin;endFill=1;"}',
        "padding": "15",
        "ignore": "id,shape,fill,stroke,refs"
    }
    csv = [f"# {key}: {value}" for key, value in graph_attributes.items()]
    csv.append("id,step,shape,fontsize,spacingbottom,refs")
    csv += subtree_to_csv(tree, None)

    return "\n".join(csv)


def subtree_to_csv(tree, parent_id):
    label = tree.name
    shape = "rectangle"
    fontsize = 11
    spacing_bottom = 0
    if isinstance(tree, BaselinesNode):
        label = "RL"
        fontsize = 32
        spacing_bottom = 20
    elif isinstance(tree, Sequence):
        label = "→"
        fontsize = 32
        spacing_bottom = 20
    elif isinstance(tree, Selector):
        label = "?"
        fontsize = 32
        spacing_bottom = 10
    elif isinstance(tree, Condition):
        shape = "ellipse"
    ref = parent_id if parent_id is not None else ""

    base_csv = [f"{str(tree.id)},{label},{shape},{fontsize},{spacing_bottom},{ref}"]

    children_csv = []
    for child in tree.children:
        child_csv = subtree_to_csv(child, str(tree.id))
        children_csv += child_csv
    return base_csv + children_csv


def save_tree_to_log(tree, filename):
    create_file_and_write(filename, lambda file: file.write(tree_to_drawio_csv(tree)))
