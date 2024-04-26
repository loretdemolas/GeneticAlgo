import concurrent
import itertools
import random
import time  # For pausing and refreshing

from matplotlib import pyplot as plt

# Task configuration
tasks = {
    "Build Factory": {"metal_cost": 620, "energy_cost": 1300, "base_build_time": 65},
    "Build Builder": {"metal_cost": 120, "energy_cost": 1750, "base_build_time": 36},
    "Build Metal Extractor": {"metal_cost": 50, "energy_cost": 500, "base_build_time": 19},
    "Build Solar Collector": {"metal_cost": 150, "energy_cost": 0, "base_build_time": 28},
    "Build Energy Converter": {"metal_cost": 1, "energy_cost": 1250, "base_build_time": 27},
    "Wait 15 Seconds": {"metal_cost": 0, "energy_cost": 0, "base_build_time": 15},
    "Build Advanced Solar Collector": {"metal_cost": 370, "energy_cost": 4000, "base_build_time": 82}
}

# Prerequisites for building tasks
task_prerequisites = {
    "Build Builder": ["Build Factory"],
    "Build Advanced Solar Collector": ["Build Builder"]
}

# Constants
MAX_METAL_EXTRACTORS = 3
DESIRED_ENERGY_METAL_RATIO = 10.0
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

    def get_build_power(self):
        return self.build_power

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
        elif task_name == "Build Advanced Solar Collector":
            self.energy_rate += 75


# Class to manage the build order and track completed tasks
class BuildOrderManager:
    def __init__(self, resource_manager, max_tasks=MAX_TASK, max_metal_extractors=MAX_METAL_EXTRACTORS):
        self.build_order_time = 0
        self.resource_manager = resource_manager
        self.max_tasks = max_tasks
        self.max_metal_extractors = max_metal_extractors
        self.build_order = []
        self.completed_tasks = set()
        self.metal_extractor_count = 0
        self.available_tasks = [
            "Build Metal Extractor",
            "Build Solar Collector",
            "Build Energy Converter",
            "Build Factory",
            "Wait 15 Seconds",
        ]

    def can_build(self, task_name):
        # If the task is "Wait 15 Seconds", always return True
        if task_name == "Wait 15 Seconds":
            return True  # No resource check needed for this task

        # Otherwise, perform the normal resource check
        task = tasks[task_name]
        current_metal = self.resource_manager.get_current_metal()
        current_energy = self.resource_manager.get_current_energy()

        return (
                current_metal >= task["metal_cost"] and
                current_energy >= task["energy_cost"]
        )

    def total_time(self, task_name):
        task = tasks[task_name]
        self.build_order_time += ((task["base_build_time"] * self.resource_manager.get_build_power())/100)

    def build_task(self, task_name):
        if not self.can_build(task_name):
            return -1  # Task can't be built
        task = tasks[task_name]
        # Check if the task is "Wait 15 Seconds"
        if task_name == "Wait 15 Seconds":
            build_time = 15  # Directly set the build time to 15 seconds
            self.total_time(task_name)
        else:
            # Calculate build time based on build power
            build_time = task["base_build_time"] / (self.resource_manager.build_power / 100)

        self.resource_manager.accumulate_resources(build_time)
        self.resource_manager.deduct_resources(task["metal_cost"], task["energy_cost"])

        self.build_order.append({"task": task_name, "build_time": build_time})
        self.completed_tasks.add(task_name)
        self.total_time(task_name)
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
                build_time = self.build_task("Wait 15 Seconds")
                if build_time == -1:
                    continue

            task_probabilities = {
                "Build Metal Extractor": 0.5,
                "Build Solar Collector": 0.4,
                "Build Builder": 0.5,
                "Build Energy Converter": 0.5,
                "Build Factory": 0.5,
                "Wait 15 Seconds": 0.1,
                "Build Advanced Solar Collector": 0.6,
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
            if build_time == -1:
                self.build_task("Wait 15 Seconds")  # Fallback to waiting
                continue

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
    def __init__(self, build_order):
        self.to_many_factories = 0
        self.advanced_building = 0
        self.waiting_time = 0
        self.metal_extractor_count = 0
        self.starting_fitness = 1000  # Base fitness
        self.build_order = build_order
        self.total_build_time = 0
        self.prerequisite_violations = False
        self.completed_tasks = set()
        self.resource_manager = ResourceManager()

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

    def simulate_build_order(self):
        # Simulate the build order and calculate resource accumulation
        build_order_manager = BuildOrderManager(self.resource_manager)
        build_time_total = 0
        waiting_time_total = 0

        for task_info in self.build_order:
            task_name = task_info["task"]

            # Simulate building the task
            build_time = build_order_manager.build_task(task_name)
            if build_time == -1:
                # Apply a small penalty for failed builds
                self.starting_fitness -= 50
                build_order_manager.build_task("Wait 15 Seconds")
                build_time = 15  # Simulated wait time
            if task_name == "Build Metal Extractor":
                self.metal_extractor_count += 1
            if task_name == "Build Advanced Solar Collector":
                self.advanced_building += 1
            if task_name == "Build Factory":
                self.to_many_factories += 1

            if task_name == "Wait 15 Seconds":
                waiting_time_total += build_time
            build_time_total += build_time

            # Simulate resource accumulation during build time
            self.resource_manager.accumulate_resources(build_time)

        self.total_build_time = build_time_total
        self.waiting_time = waiting_time_total

    def calculate_fitness(self):
        if not self.prerequisite_check():
            # Penalty for not meeting prerequisites
            self.starting_fitness -= 100

        self.simulate_build_order()  # Simulate the build order to get data

        # Calculate the current energy-to-metal ratio
        metal_rate = self.resource_manager.get_current_metal_rate()
        energy_rate = self.resource_manager.get_current_energy_rate()

        if metal_rate == 0:
            ratio = float("inf")  # Infinite ratio if no metal production
        else:
            ratio = energy_rate / metal_rate

        # Calculate ratio deviation
        ratio_penalty = abs(DESIRED_ENERGY_METAL_RATIO - ratio) * 50  # Scale penalty

        # Metal extractor penalties
        if self.metal_extractor_count > MAX_METAL_EXTRACTORS:
            # Penalty for exceeding the maximum extractors
            metal_extractor_penalty = (self.metal_extractor_count - MAX_METAL_EXTRACTORS) * 50
        elif self.metal_extractor_count < MAX_METAL_EXTRACTORS:
            # Penalty for having fewer than the maximum extractors
            metal_extractor_penalty = (MAX_METAL_EXTRACTORS - self.metal_extractor_count) * 25
        else:
            metal_extractor_penalty = 0

        # Fitness calculation with ratio penalty
        resource_penalty = self.waiting_time  # Encourage shorter build times
        tech_bonus = self.advanced_building * 50  # Encourage better builds
        to_many_factories = (self.to_many_factories - 1) * 50
        # Penalty for prerequisite violations
        illegal_move_fitness = 0
        if self.prerequisite_violations:
            illegal_move_fitness = 100
        fitness = (
                self.starting_fitness
                - ratio_penalty
                - metal_extractor_penalty
                - resource_penalty
                + tech_bonus
                - to_many_factories
                - illegal_move_fitness
        )

        extra_stats = {
            "total_build_time": self.total_build_time,
            "energy_rate": energy_rate,
            "metal_rate": metal_rate,
            "energy_metal_ratio": ratio,
            "ratio_penalty": ratio_penalty,
            "metal_extractor_penalty": metal_extractor_penalty,
            "resource_penalty": resource_penalty,
            "Tech_bonus": tech_bonus,
            "to_many_factories": to_many_factories,
            "illegal_move_fitness": illegal_move_fitness
        }

        return fitness, extra_stats

    def get_fitness(self):
        return self.calculate_fitness()[0]


def random_build_order():
    resource_manager = ResourceManager()
    build_order_manager = BuildOrderManager(resource_manager)
    return build_order_manager.create_build_order()


def crossover(parent1, parent2):
    if len(parent1) <= 1 or len(parent2) <= 1:
        return parent1, parent2

    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    if not BuildOrderFitness(child1).prerequisite_check() or not BuildOrderFitness(child2).prerequisite_check():
        return parent1, parent2

    return child1, child2


class GeneticAlgorithm:
    def __init__(self, generations, population_size, tournament_size, mutation_rate):
        self.generations = generations
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_per_generation = []
        # Populate the initial population
        for _ in range(self.population_size):
            self.population.append(random_build_order())

    def get_best_build_order(self):
        best_order = max(self.population, key=lambda bo: BuildOrderFitness(bo).get_fitness())
        best_fitness, best_stats = BuildOrderFitness(best_order).calculate_fitness()
        return best_order, best_fitness, best_stats

    def tournament_selection(self):
        selected = random.sample(self.population, self.tournament_size)
        return max(selected, key=lambda bo: BuildOrderFitness(bo).get_fitness())

    def mutation(self, chromosome):
        if len(chromosome) <= 1:
            return chromosome

        if random.random() < self.mutation_rate:
            idx1 = random.randint(0, len(chromosome) - 1)
            idx2 = random.randint(0, len(chromosome) - 1)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

        return chromosome

    def run(self):
        best_order = []
        best_fitness = -1
        best_stats = {}

        # Interactive mode to enable real-time updates
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("Best Fitness over Generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")

        for generation in range(self.generations):
            new_population = []

            # Find the best order and add to the new population
            best_order, best_fitness, best_stats = self.get_best_build_order()
            self.fitness_per_generation.append(best_fitness)  # Store best fitness
            new_population.append(best_order)

            ax.cla()  # Clear the existing plot
            ax.set_title("Best Fitness over Generations")  # Re-apply the title, labels
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best Fitness")
            # Plot all fitness values up to the current generation
            ax.plot(range(len(self.fitness_per_generation)), self.fitness_per_generation, 'bo-', markersize=5)

            # Redraw and pause to ensure the plot updates
            plt.draw()
            plt.pause(0.01)

            # Create the rest of the population through crossover and mutation
            for _ in range((self.population_size - 1) // 2):
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))

            self.population = new_population

            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")

            if best_fitness == 1000:  # Early stop condition
                break

        # End interactive mode
        plt.ioff()
        plt.show()

        return best_order, best_fitness, best_stats


# Function to perform parameter tuning
class ParameterTuning:
    def __init__(self, generations, population_sizes, tournament_sizes, mutation_rates):
        # Accept ranges for parameters to be tested
        self.generations = generations  # Total generations to run
        self.population_sizes = population_sizes  # List of population sizes to test
        self.tournament_sizes = tournament_sizes  # List of tournament sizes to test
        self.mutation_rates = mutation_rates  # List of mutation rates to test

        # To store results of parameter tuning
        self.results = []

    def tune_parameters(self):
        for population_size in self.population_sizes:
            for tournament_size in self.tournament_sizes:
                for mutation_rate in self.mutation_rates:
                    # Run the Genetic Algorithm with the current parameters
                    ga = GeneticAlgorithm(
                        generations=self.generations,
                        population_size=population_size,
                        tournament_size=tournament_size,
                        mutation_rate=mutation_rate,
                    )

                    best_order, best_fitness, best_stats = ga.run()

                    # Collect results for analysis
                    self.results.append({
                        "population_size": population_size,
                        "tournament_size": tournament_size,
                        "mutation_rate": mutation_rate,
                        "best_fitness": best_fitness,
                        "best_order": best_order,
                        "best_stats": best_stats,
                    })

    def get_best_configuration(self):
        # Return the parameter configuration with the highest fitness
        return max(self.results, key=lambda result: result["best_fitness"])


RUN_GENETIC_ALGORITHM = True  # Set to True to run the genetic algorithm

if __name__ == "__main__":
    if RUN_GENETIC_ALGORITHM:
        # Run the genetic algorithm
        ga = GeneticAlgorithm(generations=1000, population_size=100, tournament_size=5, mutation_rate=0.5)
        best_order, best_fitness, best_stats = ga.run()

        # Print the best build order
        print("Best Build Order:")
        for task in best_order:
            task_name = task["task"]
            build_time = task.get("build_time", "N/A")
            print(f"- {task_name}, Build Time: {build_time:.2f} seconds")

        # Print the best stats
        print("\nBest Stats:")
        for key, value in best_stats.items():
            print(f"{key}: {value}")

        # Plot the best fitness over generations
        plt.plot(range(len(ga.fitness_per_generation)), ga.fitness_per_generation, 'bo-', markersize=5)
        plt.title("Best Fitness over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.show()
    else:
        # Define ranges for parameter tuning
        generations = 1000
        population_sizes = [10, 25, 50, 100]
        tournament_sizes = [3, 5, 7]
        mutation_rates = [0.1, 0.3, 0.5, 0.7]

        # Initialize the parameter tuning class
        tuner = ParameterTuning(generations, population_sizes, tournament_sizes, mutation_rates)

        # Perform parameter tuning
        tuner.tune_parameters()

        # Get the best configuration
        best_configuration = tuner.get_best_configuration()

        print("Best Configuration:")
        for key, value in best_configuration.items():
            if key != "best_order":  # Avoid printing the entire build order here
                print(f"{key}: {value}")

        print("\nBest Build Order:")
        for task in best_configuration["best_order"]:
            task_name = task["task"]
            build_time = task.get("build_time", "N/A")
            print(f"- {task_name}, Build Time: {build_time:.2f} seconds")

        # Optional: Plot the best fitness over generations for the best configuration
        plt.plot(range(len(tuner.results)), [r["best_fitness"] for r in tuner.results], 'bo-', markersize=5)
        plt.title("Best Fitness for Different Configurations")
        plt.xlabel("Configuration Index")
        plt.ylabel("Best Fitness")
        plt.show()



