import concurrent
import itertools
import random

# Task configuration
tasks = {
    "Build Factory": {"metal_cost": 620, "energy_cost": 1300, "base_build_time": 65},
    "Build Builder": {"metal_cost": 120, "energy_cost": 1750, "base_build_time": 36},
    "Build Metal Extractor": {"metal_cost": 50, "energy_cost": 500, "base_build_time": 19},
    "Build Solar Collector": {"metal_cost": 150, "energy_cost": 0, "base_build_time": 28},
    "Build Energy Converter": {"metal_cost": 1, "energy_cost": 1250, "base_build_time": 27},
    "Wait 15 Seconds": {"metal_cost": 0, "energy_cost": 0, "base_build_time": 15},
}

# Prerequisites for building tasks
task_prerequisites = {
    "Build Builder": ["Build Factory"],
}

# Constants
MAX_METAL_EXTRACTORS = 3
METAL_RATE_GOAL = 60
ENERGY_RATE_GOAL = 600
METAL_ENERGY_RATIO = METAL_RATE_GOAL / ENERGY_RATE_GOAL
MAX_TASK = 100


# Class to manage resources
class ResourceManager:
    def __init__(self, initial_metal=1000, initial_energy=1000, build_power=400):
        self.current_metal = initial_metal
        self.current_energy = initial_energy
        self.build_power = build_power
        self.metal_rate = 0
        self.energy_rate = 20

    def accumulate_resources(self, time_interval):
        self.current_metal += self.metal_rate * time_interval
        self.current_energy += self.energy_rate * time_interval

    def deduct_resources(self, metal, energy):
        self.current_metal -= metal
        self.current_energy -= energy

    def get_current_metal(self):
        return self.current_metal

    def get_current_energy(self):
        return self.current_energy

    def get_current_metal_rate(self):
        return self.metal_rate

    def get_current_energy_rate(self):
        return self.energy_rate

    def update_rates(self, task_name):
        if task_name == "Build Metal Extractor":
            self.metal_rate += 1.8
        elif task_name == "Build Solar Collector":
            self.energy_rate += 20
        elif task_name == "Build Energy Converter":
            self.energy_rate -= 70
            self.metal_rate += 1
        elif task_name == "Build Builder":
            self.build_power += 100


# Class to manage the build order and track completed tasks
class BuildOrderManager:
    def __init__(self, resource_manager, max_tasks=MAX_TASK, max_metal_extractors=MAX_METAL_EXTRACTORS):
        self.resource_manager = resource_manager
        self.max_tasks = max_tasks
        self.max_metal_extractors = max_metal_extractors
        self.build_order = []
        self.completed_tasks = set()
        self.metal_extractor_count = 0
        self.available_tasks = [
            "Build Metal Extractor",
            "Build Solar Collector",
            "Build Builder",
            "Build Energy Converter",
            "Build Factory",
            "Wait 15 Seconds",
        ]

    def can_build(self, task_name):
        task = tasks[task_name]
        return (
                self.resource_manager.get_current_metal() >= task["metal_cost"]
                and self.resource_manager.get_current_energy() >= task["energy_cost"]
        )

    def build_task(self, task_name):
        if not self.can_build(task_name):
            return -1  # Task can't be built

        task = tasks[task_name]
        build_time = task["base_build_time"] / (self.resource_manager.build_power / 100)

        self.resource_manager.accumulate_resources(build_time)
        self.resource_manager.deduct_resources(task["metal_cost"], task["energy_cost"])

        self.build_order.append({"task": task_name, "build_time": build_time})
        self.completed_tasks.add(task_name)

        self.resource_manager.update_rates(task_name)

        if task_name == "Build Metal Extractor":
            self.metal_extractor_count += 1

        return build_time

    def create_build_order(self):
        while len(self.build_order) < self.max_tasks:
            # Determine valid tasks
            valid_tasks = [task for task in self.available_tasks if self.can_build(task)]

            if not valid_tasks:
                # If no valid tasks, wait and continue
                self.build_task("Wait 15 Seconds")
                continue

            task_probabilities = {
                "Build Metal Extractor": 0.5,
                "Build Solar Collector": 0.5,
                "Build Builder": 0.5,
                "Build Energy Converter": 0.5,
                "Build Factory": 0.5,
                "Wait 15 Seconds": 0.5,
            }

            extractor_cap = self.metal_extractor_count >= self.max_metal_extractors

            if extractor_cap:
                task_probabilities["Build Metal Extractor"] = 0.0  # No more metal extractors

            chosen_task = random.choices(
                valid_tasks,
                weights=[task_probabilities[task] for task in valid_tasks],
                k=1,
            )[0]

            build_time = self.build_task(chosen_task)

            # Add task prerequisites if applicable
            for key, prerequisites in task_prerequisites.items():
                if all(
                        prerequisite in self.completed_tasks
                        for prerequisite in prerequisites
                ):
                    if key not in self.available_tasks:
                        self.available_tasks.append(key)

        return self.build_order

    def total_build_time(self):
        # Sum up all the build times in the build order
        return sum(task["build_time"] for task in self.build_order)


class BuildOrderFitness:
    def __init__(self, build_order,):
        self.efficiency_penalty = None
        self.metal_fitness = None
        self.energy_fitness = None
        self.goal_met_index = None
        self.build_order = build_order
        self.completed_tasks = set()
        self.final_fitness = 0

    def prerequisite_check(self):
        for task_info in self.build_order:
            task_name = task_info["task"]
            if task_name in task_prerequisites:
                prerequisites_met = all(
                    prereq in self.completed_tasks for prereq in task_prerequisites[task_name]
                )
                if not prerequisites_met:
                    return False
            self.completed_tasks.add(task_name)
        return True

    def calculate_fitness(self, energy_production=None, metal_production=None):

        # Ensure prerequisites are met before proceeding
        if not self.prerequisite_check():
            return -1, {}

        resource_manager = ResourceManager()
        build_order_manager = BuildOrderManager(resource_manager)
        build_queue = self.build_order

        goal_met = False

        for index, task_info in enumerate(build_queue):
            task_name = task_info["task"]
            build_time = build_order_manager.build_task(task_name)
            if build_time == -1:
                return -1, {}
            metal_rate = resource_manager.get_current_metal_rate()
            energy_rate = resource_manager.get_current_energy_rate()

            if not goal_met and metal_rate >= METAL_RATE_GOAL and energy_rate >= ENERGY_RATE_GOAL:
                self.goal_met_index = index
                goal_met = True

        # Compute the energy fitness
        if resource_manager.get_current_energy_rate() >= ENERGY_RATE_GOAL:
            energy_production += 500
        energy_penalty = max(resource_manager.get_current_energy_rate() - ENERGY_RATE_GOAL*1.5, 0)
        self.energy_fitness = energy_production - energy_penalty

        # Compute the metal fitness
        if resource_manager.get_current_metal_rate() >= METAL_RATE_GOAL:
            metal_production += 500
        metal_penalty = max(resource_manager.get_current_metal_rate() - METAL_RATE_GOAL, 0)
        self.metal_fitness = metal_production - metal_penalty

        self.efficiency_penalty = self.goal_met_index

        self.final_fitness = self.energy_fitness + self.metal_fitness - self.efficiency_penalty

        if build_order_manager.metal_extractor_count >= build_order_manager.max_metal_extractors:
            self.final_fitness = 0

        # Add extra stats for debugging or additional insights
        extra_stats = {
            "energy_produced": resource_manager.get_current_energy(),
            "metal_produced": resource_manager.get_current_metal(),
            "total_time": build_order_manager.total_build_time(),
        }

        return self.final_fitness, extra_stats

    def get_fitness(self):
        return self.calculate_fitness()[0]


class GeneticAlgorithm:
    def __init__(self, generations, population_size, tournament_size, mutation_rate):
        self.generations = generations
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.population = [self.random_build_order() for _ in range(population_size)]

    def random_build_order(self):
        resource_manager = ResourceManager()
        build_order_manager = BuildOrderManager(resource_manager)
        return build_order_manager.create_build_order()

    def get_best_build_order(self):
        best_order = max(self.population, key=lambda bo: BuildOrderFitness(bo).get_fitness())
        best_fitness, best_stats = BuildOrderFitness(best_order).calculate_fitness()
        return best_order, best_fitness, best_stats

    def tournament_selection(self):
        selected = random.sample(self.population, self.tournament_size)
        return max(selected, key=lambda bo: BuildOrderFitness(bo).get_fitness())

    def crossover(self, parent1, parent2):
        if len(parent1) <= 1 or len(parent2) <= 1:
            return parent1, parent2

        point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        if not BuildOrderFitness(child1).prerequisite_check() or not BuildOrderFitness(child2).prerequisite_check():
            return parent1, parent2

        return child1, child2

    def mutation(self, chromosome):
        if len(chromosome) <= 1:
            return chromosome

        if random.random() < self.mutation_rate:
            idx1 = random.randint(0, len(chromosome) - 1)
            idx2 = random.randint(0, len(chromosome) - 1)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

        return chromosome

    def run(self):
        best_build_order = None
        best_fitness = -1
        best_stats = {}

        for generation in range(self.generations):
            new_population = []

            # Find best order and add to new population
            best_order, best_fitness, best_stats = self.get_best_build_order()
            new_population.append(best_order)

            # Create rest of the population through crossover and mutation
            for _ in range((self.population_size - 1) // 2):
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))

            self.population = new_population

            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")

            if best_fitness > 800:
                break

        if best_fitness < 0:  # No valid build orders found
            raise RuntimeError("No valid build order found")

        return best_build_order, best_fitness, best_stats


# Function to perform parameter tuning

RUN_GENETIC_ALGORITHM = True  # Set to True to run the genetic algorithm

if __name__ == "__main__":
    if RUN_GENETIC_ALGORITHM:
        ga = GeneticAlgorithm(generations=2000, population_size=50, tournament_size=5, mutation_rate=0.5)
        best_order, best_fitness, best_stats = ga.run()

        print("Optimal build order:", best_order)
        print("Best fitness:", best_fitness)
        print("Additional stats:", best_stats)

