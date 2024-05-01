
import random

from task_and_constants import tasks, task_prerequisites, MAX_METAL_EXTRACTORS


class BuildOrderManager:
    def __init__(self, resource_manager, max_tasks=100, max_metal_extractors=MAX_METAL_EXTRACTORS):
        self.resource_manager = resource_manager
        self.max_tasks = max_tasks
        self.max_metal_extractors = max_metal_extractors
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

        # Deduct resources and accumulate them during build
        self.resource_manager.deduct_resources(task["metal_cost"], task["energy_cost"])
        self.resource_manager.accumulate_resources(build_time)

        # Update build order
        self.build_order.append(
            {
                "task": task_name,
                "build_time": build_time,
                "current_metal": self.resource_manager.get_current_metal(),
                "current_energy": self.resource_manager.get_current_energy(),
            }
        )

        self.completed_tasks.add(task_name)
        self.resource_manager.update_rates(task_name)

        # Handle special tasks (existing logic)
        if task_name == "Build Factory":
            self.built_factory = True
        elif task_name == "Build Advanced Factory":
            self.built_adv_factory = True
        elif task_name == "Build Advanced Builder":
            self.built_adv_builder = True
        elif task_name == "Build Metal Extractor":
            self.metal_extractor_count += 1
        elif task_name == "Build Advanced Metal Extractor":
            self.advanced_metal_extractor_count += 1

        return build_time

    def create_build_order(self):
        while len(self.build_order) < self.max_tasks:
            valid_tasks = [
                task for task in self.available_tasks if self.can_build(task)
            ]

            if not valid_tasks:
                # Wait if no valid tasks
                self.build_task("Wait 15 Seconds")
                continue

            # Probabilities for each task
            task_probabilities = {
                "Build Metal Extractor": 0.2,
                "Build Solar Collector": 0.2,
                "Build Builder": 0.2,
                "Build Energy Converter": 0.1,
                "Build Factory": 0.1,
                "Build Advanced Solar Collector": 0.2,
                "Build Advanced Factory": 0.5,
                "Build Advanced Builder": 0.4,
                "Build Advanced Metal Extractor": 0.4,
                "Build Advanced Energy Converter": 0.3,
                "Wait 15 Seconds": 0.01,
            }

            # Adjust probabilities based on current conditions
            if self.built_factory:
                task_probabilities["Build Factory"] = 0

            if self.built_adv_factory:
                task_probabilities["Build Advanced Factory"] = 0

            if not self.built_adv_builder:
                task_probabilities["Build Advanced Builder"] += 0.3

            if self.metal_extractor_count >= self.max_metal_extractors:
                task_probabilities["Build Metal Extractor"] = 0

            if self.advanced_metal_extractor_count >= self.max_metal_extractors:
                task_probabilities["Build Advanced Metal Extractor"] = 0

            if self.built_adv_solar:
                task_probabilities["Build Solar Collector"] = 0

            # Choose a task based on probabilities
            chosen_task = random.choices(
                valid_tasks,
                weights=[task_probabilities[task] for task in valid_tasks],
                k=1,
            )[0]

            build_time = self.build_task(chosen_task)

            if build_time == -1:
                # Wait if build failed
                self.build_task("Wait 15 Seconds")
                continue

            # Add prerequisites for future tasks
            for key, prerequisites in task_prerequisites.items():
                if all(prereq in self.completed_tasks for prereq in prerequisites):
                    if key not in self.available_tasks:
                        self.available_tasks.append(key)

        return self.build_order
