import os

from malmo import malmoutils

from agent import MalmoAgent
from runner import Runner


def run():
    malmoutils.fix_print()

    agent = MalmoAgent()
    runner = Runner(agent)
    runner.run_mission()


if __name__ == "__main__":
    run()
