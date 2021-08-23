import os

from malmo import malmoutils

from agent import MalmoAgent
from runner import Runner


def run():
    if "MALMO_XSD_PATH" not in os.environ:
        print("Please set the MALMO_XSD_PATH environment variable.")
        return
    malmoutils.fix_print()

    agent = MalmoAgent()
    runner = Runner(agent)
    runner.run_mission()


if __name__ == "__main__":
    run()
