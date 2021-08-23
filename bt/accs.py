from py_trees.composites import Selector

from bt.actions import Action
from bt.conditions import Condition
from bt.sequence import Sequence


def find_accs(node):
    current = node
    path = []
    while current is not None:
        path.append(current)
        current = current.parent
    path = path[::-1]
    n0s = []
    for pnode in path:
        if not isinstance(current, Action) or pnode != node.parent:
            if isinstance(pnode, Sequence):
                n0s.append(pnode)
    n1 = []
    for n0 in n0s:
        for child in n0.children:
            if child in path:
                break
            n1.append(child)

    n1c = []
    n2c = []
    for nnode in n1:
        if isinstance(nnode, Condition):
            n1c.append(nnode)
        elif is_selector(nnode):
            n2c.append(nnode.children[0])

    return n1c + n2c


def is_selector(node):
    return isinstance(node, Selector) and not isinstance(node, Sequence)


def test_acc():
    # Arrange: Create the ACC example from Petter's paper
    in_safe_area_ppa_pre_conditions = Sequence(
        "Move to safe Area PPA PreCons",
        children=[Condition("Free Path to Safe Area Exists"), Action("Move to Safe Area")]
    )
    in_safe_area_condition = Condition("In Safe Area")
    in_safe_area_ppa_post_conditions = Selector(
        "Move to Safe Area PPA PostCons",
        children=[in_safe_area_condition, in_safe_area_ppa_pre_conditions]
    )
    robot_near_object_ppa_pre_conditions = Sequence(
        "Robot Near Object PPA PreCons",
        children=[Condition("Free Path To Object exists"), Action("Move to Object")]
    )
    robot_near_object_ppa_post_conditions = Selector(
        "Robot Near Object Area PPA PostCons",
        children=[Condition("Robot Near Object"), robot_near_object_ppa_pre_conditions]
    )
    object_in_gripper_ppa_pre_conditions = Sequence(
        "Object in Gripper PPA PreCons",
        children=[robot_near_object_ppa_post_conditions, Action("Grasp Object")]
    )
    object_in_gripper_condition = Condition("Object in Gripper")
    object_in_gripper_ppa_post_conditions = Selector(
        "Object in Gripper PPA PostCons",
        children=[object_in_gripper_condition, object_in_gripper_ppa_pre_conditions]
    )

    move_to_goal_action = Action("Move To Goal")
    close_to_goal_ppa_pre_conditions = Sequence(
        "Close To Goal Object PPA PreCons",
        children=[Condition("Free Path to Goal Exists"), move_to_goal_action]
    )
    close_to_goal_ppa_post_conditions = Selector(
        "Close to Goal Area PPA PostCons",
        children=[Condition("Close to Goal"), close_to_goal_ppa_pre_conditions]
    )
    robot_has_cash_ppa_pre_conditions = Sequence(
        "Robot has Cash PPA PostCons",
        children=[Condition("Payed task available"), Action("Do task and earn cash")]
    )
    robot_has_cash_ppa_post_conditions = Selector(
        "Robot has Cash Area PPA PostCons",
        children=[Condition("Robot has cash"), robot_has_cash_ppa_pre_conditions]
    )
    object_at_goal_preconditions_one = Sequence(
        "Object at Goal PPA PreCons 1",
        children=[
            object_in_gripper_ppa_post_conditions,
            close_to_goal_ppa_post_conditions,
            Action("Place object at goal")
        ]
    )
    object_at_goal_preconditions_two = Sequence(
        "Object at Goal PPA PreCons 2",
        children=[
            Condition("Agent nearby"),
            robot_has_cash_ppa_post_conditions,
            Action("Pay agent to place object")
        ]
    )
    object_at_goal_post_conditions = Selector(
        "Object At Goal PPA PostCons",
        children=[Condition("Object at Goal"), object_at_goal_preconditions_one, object_at_goal_preconditions_two]
    )
    main_sequence = Sequence(
        "Main Sequence",
        children=[in_safe_area_ppa_post_conditions, object_at_goal_post_conditions]
    )
    main_sequence.setup_with_descendants()

    # Act:
    accs = find_accs(move_to_goal_action)

    # Assert:
    assert in_safe_area_condition in accs
    assert object_in_gripper_condition in accs


if __name__ == "__main__":
    test_acc()
