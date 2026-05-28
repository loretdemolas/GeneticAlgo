
from resource_manager import ResourceManager
from build_order_manager import BuildOrderManager
from task_and_constants import tasks, task_prerequisites, DESIRED_ENERGY_METAL_RATIO, MAX_METAL_EXTRACTORS


class BuildOrderFitness:
    def __init__(self, build_order):
        self.adv_builder = False
        self.adv_metal_extractor = False
        self.adv_factories = False
        self.adv_solar_count = 0
        self.metal_adv_extractor_count = 0
        self.too_many_adv_factories = 0
        self.too_many_factories = 0
        self.to_many_factories = 0
        self.advanced_building = 0
        self.waiting_time = 0
        self.metal_extractor_count = 0
        self.starting_fitness = 1000  # Base fitness
        self.build_order = build_order
        self.total_build_time = 0
        self.unique_task_bonus = 0
        self.prerequisite_violations = False
        self.completed_tasks = set()
        self.resource_manager = ResourceManager()
        self.task_count = {}

    def simulate_build_order(self):
        build_order_manager = BuildOrderManager(self.resource_manager)
        build_time_total = 0
        waiting_time_total = 0

        for task_info in self.build_order:
            task_name = task_info["task"]
            build_time = build_order_manager.build_task(task_name)

            if build_time == -1:
                self.starting_fitness -= 50  # Penalty for failed builds
                build_order_manager.build_task("Wait 15 Seconds")
                build_time = 15

            build_time_total += build_time
            if task_name == "Wait 15 Seconds":
                waiting_time_total += build_time

            # More simulation logic...

        self.total_build_time = build_time_total
        self.waiting_time = waiting_time_total

    def prerequisite_check(self):
        # Ensure prerequisites are met in the build order
        for task_info in self.build_order:
            task_name = task_info["task"]
            if task_name in task_prerequisites:
                prerequisites_met = all(
                    prereq in self.completed_tasks for prereq in task_prerequisites[task_name]
                )
                if not prerequisites_met:
                    self.prerequisite_violations = True
                    return False
            self.completed_tasks.add(task_name)

        return True

    def calculate_unique_task_bonus(self):
        # Calculate the bonus based on how many unique tasks are in the build order
        unique_tasks = set(task_info["task"] for task_info in self.build_order)
        return len(unique_tasks) * 20  # Bonus per unique task

    def calculate_high_cost_bonus(self):
        # Calculate bonus based on the total metal and energy cost of tasks in the build order
        total_metal_cost = 0
        total_energy_cost = 0

        for task_info in self.build_order:
            task_name = task_info["task"]
            task = tasks[task_name]
            total_metal_cost += task["metal_cost"]
            total_energy_cost += task["energy_cost"]

        # Bonus proportional to the total costs
        high_cost_bonus = (total_metal_cost / 10) + (total_energy_cost / 100)
        return high_cost_bonus

    def calculate_penalties(self):
        # Calculate penalties for various build order aspects
        metal_rate = self.resource_manager.get_current_metal_rate()
        energy_rate = self.resource_manager.get_current_energy_rate()

        # Penalty for deviation from desired energy-metal ratio
        ratio_penalty = 0
        if metal_rate > 0:
            ratio = energy_rate / metal_rate
            ratio_penalty = abs(ratio - DESIRED_ENERGY_METAL_RATIO) * 50

        # Penalty for too many or too few metal extractors
        metal_extractor_penalty = 0
        if self.metal_extractor_count > MAX_METAL_EXTRACTORS:
            metal_extractor_penalty += (self.metal_extractor_count - MAX_METAL_EXTRACTORS) * 50

        if self.metal_adv_extractor_count > MAX_METAL_EXTRACTORS:
            metal_extractor_penalty += (self.metal_adv_extractor_count - MAX_METAL_EXTRACTORS) * 50

        # Penalty for waiting too long
        waiting_penalty = max(self.waiting_time / 2, 0)

        # Penalty for too many factories
        factory_penalty = max((self.too_many_factories - 1) * 50, 0)

        # Penalty for prerequisite violations
        illegal_move_penalty = 100 if self.prerequisite_violations else 0

        # Aggregate all penalties
        total_penalties = (
                ratio_penalty
                + metal_extractor_penalty
                + waiting_penalty
                + factory_penalty
                + illegal_move_penalty
        )

        return total_penalties

    def calculate_fitness(self):
        if not self.prerequisite_check():
            self.starting_fitness -= 100

        self.simulate_build_order()

        fitness = self.starting_fitness
        fitness += self.calculate_unique_task_bonus()  # Unique task bonus
        fitness += self.calculate_high_cost_bonus()  # High-cost bonus

        # Subtract total penalties
        fitness -= self.calculate_penalties()

        extra_stats = {
            "total_build_time": self.total_build_time,
            "energy_rate": self.resource_manager.get_current_energy_rate(),
            "metal_rate": self.resource_manager.get_current_metal_rate(),
            "energy_metal_ratio":
                self.resource_manager.get_current_energy_rate() / self.resource_manager.get_current_metal_rate(),
            "waiting_penalty": max(self.waiting_time / 2, 0),
            "Tech_bonus": self.advanced_building * 100,
            "factory_penalty": max((self.too_many_factories - 1) * 50, 0),
            "metal_extractor_penalty": (self.metal_extractor_count - MAX_METAL_EXTRACTORS) * 50,
            "illegal_move_penalty": 100 if self.prerequisite_violations else 0,
        }

        return fitness, extra_stats

    def get_fitness(self):
        return self.calculate_fitness()[0]

