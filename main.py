import os
from malmo import malmoutils

from agent import BaseAgent
from runner import Runner


def run():
    if "MALMO_XSD_PATH" not in os.environ:
        print("Please set the MALMO_XSD_PATH environment variable.")
        return
    malmoutils.fix_print()

    agent = BaseAgent()
    runner = Runner(agent)
    runner.run_mission()


if __name__ == "__main__":
    run()
