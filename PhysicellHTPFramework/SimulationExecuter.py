import os
from typing import *
import numpy as np
import pandas as pd
import cmd
import SettingsConfigurator
import json


class PhysicellSimulationExecutor:
    def __init__(self, physicell_dir: str):
        self.physicell_dir = physicell_dir
