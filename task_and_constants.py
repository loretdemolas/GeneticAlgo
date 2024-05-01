# tasks_and_constants.py
# Task configuration
tasks = {
    "Build Factory": {"metal_cost": 620, "energy_cost": 1300, "base_build_time": 65},
    "Build Builder": {"metal_cost": 120, "energy_cost": 1750, "base_build_time": 36},
    "Build Metal Extractor": {"metal_cost": 50, "energy_cost": 500, "base_build_time": 19},
    "Build Solar Collector": {"metal_cost": 150, "energy_cost": 0, "base_build_time": 28},
    "Build Energy Converter": {"metal_cost": 1, "energy_cost": 1250, "base_build_time": 27},
    "Wait 15 Seconds": {"metal_cost": 0, "energy_cost": 0, "base_build_time": 15},
    "Build Advanced Solar Collector": {"metal_cost": 370, "energy_cost": 4000, "base_build_time": 82},
    "Build Advanced Factory": {"metal_cost": 2900, "energy_cost": 16000, "base_build_time": 168},
    "Build Advanced Builder": {"metal_cost": 470, "energy_cost": 6900, "base_build_time": 97},
    "Build Advanced Metal Extractor": {"metal_cost": 640, "energy_cost": 8100, "base_build_time": 141},
    "Build Advanced Energy Converter": {"metal_cost": 370, "energy_cost": 21000, "base_build_time": 27},
}

# Prerequisites for building tasks
task_prerequisites = {
    "Build Builder": ["Build Factory"],
    "Build Advanced Solar Collector": ["Build Builder"],
    "Build Advanced Factory": ["Build Builder"],
    "Build Advanced Builder": ["Build Advanced Factory"],
    "Build Advanced Metal Extractor": ["Build Advanced Builder"],
    "Build Advanced Energy Converter": ["Build Advanced Builder"],
}

# Constants
MAX_METAL_EXTRACTORS = 6
DESIRED_ENERGY_METAL_RATIO = 10.0
MAX_TASK = 100
