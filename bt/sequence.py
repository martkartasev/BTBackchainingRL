import py_trees as pt

class Sequence(pt.composites.Selector):
    """
    Borrowed from https://github.com/smarc-project/smarc_missions/blob/noetic-devel/smarc_bt/src/bt_common.py
    For testing purposes.
    Reactive sequence overriding sequence with memory, py_trees' only available sequence.
    """

    def __init__(self, name="Sequence", children=None):
        super(Sequence, self).__init__(name=name, children=children)

    def tick(self):
        """
        Run the tick behaviour for this selector. Note that the status
        of the tick is always determined by its children, not
        by the user customised update function.
        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # Required behaviour for *all* behaviours and composites is
        # for tick() to check if it isn't running and initialise
        if self.status != pt.common.Status.RUNNING:
            # selectors don't do anything specific on initialisation
            #   - the current child is managed by the update, never needs to be 'initialised'
            # run subclass (user) handles
            self.initialise()
        # run any work designated by a customised instance of this class
        self.update()
        previous = self.current_child
        for child in self.children:
            for node in child.tick():
                yield node
                if node is child:
                    if node.status == pt.common.Status.RUNNING or node.status == pt.common.Status.FAILURE:
                        self.current_child = child
                        self.status = node.status
                        if previous is None or previous != self.current_child:
                            # we interrupted, invalidate everything at a lower priority
                            passed = False
                            for child in self.children:
                                if passed:
                                    if child.status != pt.common.Status.INVALID:
                                        child.stop(pt.common.Status.INVALID)
                                passed = True if child == self.current_child else passed
                        yield self
                        return
        # all children succeeded, set succeed ourselves and current child to the last bugger who failed us
        self.status = pt.common.Status.SUCCESS
        try:
            self.current_child = self.children[-1]
        except IndexError:
            self.current_child = None
        yield self
