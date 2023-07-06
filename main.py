import sys
import time

from runner import run


def main():
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = int(time.time())
    run(experiment_name)


if __name__ == "__main__":
    main()
