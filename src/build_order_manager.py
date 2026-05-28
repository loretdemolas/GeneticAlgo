import random

import resource_manager
from task_and_constants import tasks, phase_one_task_prerequisites, MAX_METAL_EXTRACTORS, PHASE_ONE_METAL_GOAL, \
    PHASE_ONE_Energy_GOAL


class BuildOrderManager:
    def __init__(self, resource_manager, max_tasks=100, ):
        self.adv_builder_count = 0
        self.metal_adv_extractor_cap_reached = False
        self.phase_two_metal_goal = None
        self.phase_two_goal = False
        self.builder_count = 0
        self.solar_count = 0
        self.metal_extractor_cap_reached = False
        self.built_builder = False
        self.phase_one_energy_goal = PHASE_ONE_Energy_GOAL
        self.phase_one_metal_goal = PHASE_ONE_METAL_GOAL
        self.phase_one_goal = False
        self.resource_manager = resource_manager
        self.max_tasks = max_tasks
        self.max_metal_extractors = MAX_METAL_EXTRACTORS
        self.build_order = []
        self.completed_tasks = set()
        self.metal_extractor_count = 0
        self.advanced_metal_extractor_count = 0
        self.energy_convert_count = 0
        self.adv_solar_count = 0
        self.built_factory = False
        self.built_adv_solar = False
        self.built_adv_factory = False  # Added for tracking
        self.built_adv_builder = False  # Added for tracking
        self.available_tasks = [
            "Build Metal Extractor",
            "Build Solar Collector",
            "Build Energy Converter",
            "Build Factory",
            "Wait 15 Seconds",
        ]

    def can_build(self, task_name):
        # Check if sufficient resources are available
        if task_name == "Wait 15 Seconds":
            return True

        task = tasks[task_name]
        current_metal = self.resource_manager.get_current_metal()
        current_energy = self.resource_manager.get_current_energy()

        return (
                current_metal >= task["metal_cost"]
                and current_energy >= task["energy_cost"]
        )

    def build_task(self, task_name):
        if not self.can_build(task_name):
            return -1  # Task can't be built

        task = tasks[task_name]
        build_power = self.resource_manager.get_build_power()

        # Build time calculation
        build_time = task["base_build_time"] / (build_power / 100)

        if task_name == "Wait 15 Seconds":
            build_time = 15.0

        # Deduct resources and accumulate them during build
        self.resource_manager.deduct_resources(task["metal_cost"], task["energy_cost"])
        self.resource_manager.accumulate_resources(build_time)

        # Dictionary representing a build task
        build_order_entry = {
            "task": task_name,
            "build_time": build_time,
            "current_metal": self.resource_manager.get_current_metal(),
            "current_energy": self.resource_manager.get_current_energy(),
        }

        # Append the build order entry to the list
        self.build_order.append(build_order_entry)

        # Print the build order entry
        print(f"Added build task: {build_order_entry}"
              f"Energy Rate: {self.resource_manager.get_current_energy_rate()}"
              f"Metal Rate: {self.resource_manager.get_current_metal_rate()}")

        self.completed_tasks.add(task_name)
        self.resource_manager.update_rates(task_name)

        # Handle special tasks (existing logic)
        if task_name == "Build Factory":
            self.built_factory = True
        elif task_name == "Build Advanced Solar Collector":
            self.built_adv_solar = True
        elif task_name == "Build Builder":
            self.built_builder = True
            self.builder_count += 1
        elif task_name == "Build Solar Collector":
            self.solar_count += 1
        elif task_name == "Build Advanced Factory":
            self.built_adv_factory = True
        elif task_name == "Build Advanced Builder":
            self.built_adv_builder = True
        elif task_name == "Build Metal Extractor":
            self.metal_extractor_count += 1
            if self.metal_extractor_count >= self.max_metal_extractors:
                self.metal_extractor_cap_reached = True
        elif task_name == "Build Advanced Metal Extractor":
            self.advanced_metal_extractor_count += 1

        return build_time

    def get_valid_tasks(self):
        valid_tasks = [task for task in self.available_tasks if self.can_build(task)]

        if not valid_tasks:
            # Wait if no valid tasks
            self.build_task("Wait 15 Seconds")

        return valid_tasks

    def update_prerequisites(self, task_pre_req):
        for key, prerequisites in task_pre_req.items():
            if all(prereq in self.completed_tasks for prereq in prerequisites):
                if key not in self.available_tasks:
                    self.available_tasks.append(key)

    def create_build_order(self):
        self.phase_one_build_order()

        return self.build_order

    def phase_one_build_order(self):
        while not self.phase_one_goal:
            valid_tasks = self.get_valid_tasks()  # Call helper function

            if not valid_tasks:
                continue

            # Probabilities for each task
            task_probabilities = {
                "Build Metal Extractor": 0.2,
                "Build Solar Collector": 0.2,
                "Build Builder": 0.3,
                "Build Energy Converter": 0.1,
                "Build Factory": 0.8,
                "Build Advanced Solar Collector": 1.2,
                "Wait 15 Seconds": 0.01,
            }
            # Phase one goals
            metal_goal_reached = self.resource_manager.get_current_metal_rate() >= self.phase_one_metal_goal

            # Adjust probabilities and mark goals based on current conditions
            if self.built_factory:
                task_probabilities["Build Factory"] = 0.0
            if self.metal_extractor_count >= self.max_metal_extractors:
                task_probabilities["Build Metal Extractor"] = 0.0
                self.metal_extractor_cap_reached = True
            if self.solar_count >= 10:
                task_probabilities["Build Solar Collector"] = 0.0
            if self.builder_count > 4:
                task_probabilities["Build Builder"] = 0.0
            if metal_goal_reached:
                task_probabilities["Build Energy Converter"] = 0.0
            if self.resource_manager.get_current_energy_rate() < 72:
                task_probabilities["Build Energy Converter"] = 0.0

            # Choose a task based on probabilities
            chosen_task = random.choices(
                valid_tasks,
                weights=[task_probabilities[task] for task in valid_tasks],
                k=1,
            )[0]
            if (self.can_build("Build Advanced Solar Collector")
                    and self.built_factory
                    and task_probabilities["Build Energy Converter"] == 0
            ):
                chosen_task = "Build Advanced Solar Collector"

            build_time = self.build_task(chosen_task)

            if build_time == -1:
                # Wait if build failed
                self.build_task("Wait 15 Seconds")
                continue

            # Add prerequisites for future tasks
            self.update_prerequisites(phase_one_task_prerequisites)
            # Check to see if Phase One energy and metal goals are reached
            metal_goal_reached = self.resource_manager.get_current_metal_rate() >= self.phase_one_metal_goal
            energy_goal_reached = self.resource_manager.get_current_energy_rate() >= self.phase_one_energy_goal
            # if Phase One goals are reached, exit loop
            if (self.built_factory
                    and self.metal_extractor_cap_reached
                    and energy_goal_reached
                    and metal_goal_reached
                    and self.built_builder
                    and self.built_adv_solar
            ):
                self.phase_one_goal = True
        return self.build_order

    def phase_two_build_order(self):
        while not self.phase_two_goal:
            valid_tasks = self.get_valid_tasks()  # Call helper function

            if not valid_tasks:
                continue

            # Probabilities for each task
            task_probabilities = {
                "Build Advanced Metal Extractor": 0.2,
                "Build Advanced Builder": 0.3,
                "Build Advanced Factory": 0.8,
                "Wait 15 Seconds": 0.01,
            }
            # Phase one goals
            metal_goal_reached = self.resource_manager.get_current_metal_rate() >= self.phase_two_metal_goal

            if self.built_adv_factory:
                task_probabilities["Build Advanced Factory"] = 0.0
            if self.advanced_metal_extractor_count >= self.max_metal_extractors:
                task_probabilities["Build Advanced Metal Extractor"] = 0.0
                self.metal_adv_extractor_cap_reached = True
            if self.adv_builder_count > 2:
                task_probabilities["Build Advanced Builder"] = 0.0
            if metal_goal_reached:
                task_probabilities["Build Advanced Energy Converter"] = 0.0
            if self.resource_manager.get_current_energy_rate() < 600:
                task_probabilities["Build Energy Converter"] = 0.0

            chosen_task = random.choices(
                valid_tasks,
                weights=[task_probabilities[task] for task in valid_tasks],
                k=1,
            )[0]
            if (self.can_build("Build Advanced Solar Collector")
                    and self.built_factory
                    and task_probabilities["Build Energy Converter"] == 0
            ):
                chosen_task = "Build Advanced Solar Collector"

            build_time = self.build_task(chosen_task)

            if build_time == -1:
                # Wait if build failed
                self.build_task("Wait 15 Seconds")
                continue

            # Add prerequisites for future tasks
            self.update_prerequisites(phase_one_task_prerequisites)
            # Check to see if Phase One energy and metal goals are reached
            metal_goal_reached = self.resource_manager.get_current_metal_rate() >= self.phase_one_metal_goal
            energy_goal_reached = self.resource_manager.get_current_energy_rate() >= self.phase_one_energy_goal
            # if Phase One goals are reached, exit loop
            if (self.built_factory
                    and self.metal_extractor_cap_reached
                    and energy_goal_reached
                    and metal_goal_reached
                    and self.built_builder
                    and self.built_adv_solar
            ):
                self.phase_one_goal = True
        return self.build_order
