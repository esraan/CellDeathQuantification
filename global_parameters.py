import os
# todo: add explanation to all global parameters


SHOWFIG = False
SAVEFIG = ~SHOWFIG


RECENT_DEATH_ONLY_FLAG = False

WINDOW_SIZE = 5

DIST_THRESHOLD_IN_PIXELS = 200


EPSILON = 1e-15

USE_LOG = True

CONFIG_FILES_DIR_PATH = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['config_files'])
