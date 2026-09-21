import os

SRC_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
)
CONFIG_PATH = os.path.join(SRC_PATH, "hmm4ppm", "configs")
EXPORT_PATH = os.path.join(SRC_PATH, "..", "export")
DATA_PATH = os.path.join(SRC_PATH, "..", "data")

os.makedirs(EXPORT_PATH, exist_ok=True)