import logging
import os

from nlpstack.commands import main

if os.environ.get("NLPSTACK_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("NLPSTACK_LOG_LEVEL", "INFO")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)


def run() -> None:
    main(prog="nlpstack")


if __name__ == "__main__":
    run()
