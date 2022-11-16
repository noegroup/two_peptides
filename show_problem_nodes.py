
from glob import glob


def show_device(f):
    with open(f, "r") as fp:
        for line in fp:
            if "GPU id" in line:
                return line


def has_error(f):
    with open(f, "r") as fp:
        return "ERROR" in fp.read()


logs = glob("log/sim_*_*.log")

for log in logs:
    if has_error(log):
        print(show_device(log))

