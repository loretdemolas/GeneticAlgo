import random
from itertools import product
import concurrent.futures
import itertools

# Define tasks with costs, build times, and names
tasks = {
    "Build Factory": {"metal_cost": 620, "energy_cost": 1300, "base_build_time": 65},
    "Build Builder": {"metal_cost": 120, "energy_cost": 1750, "base_build_time": 36},
    "Build Metal Extractor": {"metal_cost": 50, "energy_cost": 500, "base_build_time": 19},
    "Build Solar Collector": {"metal_cost": 150, "energy_cost": 0, "base_build_time": 28},
    "Build Energy Converter": {"metal_cost": 1, "energy_cost": 1250, "base_build_time": 27},
    "Wait 15 Seconds": {"metal_cost": 0, "energy_cost": 0, "base_build_time": 15},
}

task_prerequisites = {
    "Build Builder": ["Build Factory"],
}


class ResourceManager:
    def __init__(self, initial_metal=1000, initial_energy=1000, build_power=400, goal_metal_rate=5.0, goal_energy_rate=100.0, max_metal_extractors=3):
        self.current_metal = initial_metal
        self.current_energy = initial_energy
        self.build_power = build_power
        self.metal_rate = 0
        self.energy_rate = 20
        self.time_elapsed = 0
        self.completed_tasks = set()
        self.goal_metal_rate = goal_metal_rate
        self.goal_energy_rate = goal_energy_rate
        self.metal_extractor_count = 0  # Counter for Metal Extractors
        self.max_metal_extractors = max_metal_extractors  # Maximum allowed Metal Extractors

    def accumulate_resources(self, time_interval):
        self.current_metal += self.metal_rate * time_interval
        self.current_energy += self.energy_rate * time_interval

    def can_build(self, task_name):
        if task_name == "Build Metal Extractor" and self.metal_extractor_count >= self.max_metal_extractors:
            return False
        task = tasks[task_name]
        return (
                self.current_metal >= task["metal_cost"]
                and self.current_energy >= task["energy_cost"]
        )

    def build_task(self, task_name):
        if not self.can_build(task_name):
            return -1  # Return -1 if the build is not allowed

        if task_name == "Wait 15 Seconds":
            build_time = 60
            self.accumulate_resources(build_time)
            self.time_elapsed += build_time
            return build_time

        task = tasks[task_name]
        build_time = task["base_build_time"] / (self.build_power / 100)

        self.accumulate_resources(build_time)

        self.current_metal -= task["metal_cost"]
        self.current_energy -= task["energy_cost"]
        self.time_elapsed += build_time

        self.update_rates(task_name)

        return build_time

    def update_rates(self, task_name):
        if task_name == "Build Metal Extractor":
            self.metal_rate += 1.8
            self.metal_extractor_count += 1
        elif task_name == "Build Solar Collector":
            self.energy_rate += 20
        elif task_name == "Build Energy Converter":
            self.energy_rate -= 70
            self.metal_rate += 1
        elif task_name == "Build Builder":
            self.build_power += 100

    def print_stats(self):
        print('################################')
        print(f'Current Metal: {self.current_metal:.2f}')
        print(f'Current Energy: {self.current_energy:.2f}')
        print(f'Build Power: {self.build_power}')
        print(f'Metal Rate: {self.metal_rate:.2f} per second')
        print(f'Energy Rate: {self.energy_rate:.2f} per second')
        print(f'Time Elapsed: {self.time_elapsed:.2f} seconds')
        print(' ')

    def is_goal_met(self):
        return (
                self.metal_rate >= self.goal_metal_rate
                and self.energy_rate >= self.goal_energy_rate
        )


def random_build_order(max_tasks, max_metal_extractors=3):
    resource_manager = ResourceManager(max_metal_extractors=max_metal_extractors)
    build_order = []
    build_index = 0

    available_tasks = [
        "Build Metal Extractor",
        "Build Solar Collector",
        "Build Builder",
        "Build Energy Converter",
        "Build Factory",
        "Wait 15 Seconds",
    ]

    for _ in range(max_tasks):
        # Determine valid tasks
        valid_tasks = [task for task in available_tasks if resource_manager.can_build(task)]

        if not valid_tasks:
            resource_manager.build_task("Wait 15 Seconds")
            build_order.append({"task": "Wait 15 Seconds", "build_index": build_index})
            build_index += 1
            continue

        # Set task probabilities based on current resource status relative to goal rates
        task_probabilities = {
            "Build Metal Extractor": 0.5,
            "Build Solar Collector": 0.5,
            "Build Energy Converter": 0.5,
            "Build Builder": 0.5,
            "Build Factory": 0.5,
            "Wait 1 Minute": 0.5,
        }

        # Adjust probabilities based on whether goal rates are met
        if resource_manager.metal_extractor_count >= max_metal_extractors:
            task_probabilities["Build Metal Extractor"] = 0.0  # No more Metal Extractors
        elif resource_manager.metal_rate < resource_manager.goal_metal_rate:
            task_probabilities["Build Metal Extractor"] = 1.0
        else:
            task_probabilities["Build Metal Extractor"] = 0.5

        if resource_manager.energy_rate < resource_manager.goal_energy_rate:
            task_probabilities["Build Solar Collector"] = 1.0
        else:
            task_probabilities["Build Solar Collector"] = 0.5

        if resource_manager.energy_rate > resource_manager.metal_rate * 6:
            task_probabilities["Build Energy Converter"] = 1.0
        else:
            task_probabilities["Build Energy Converter"] = 0.5

        chosen_task = random.choices(
            valid_tasks,
            weights=[task_probabilities[task] for task in valid_tasks]
        )[0]

        build_time = resource_manager.build_task(chosen_task)
        build_order.append({"task": chosen_task, "build_index": build_index})
        build_index += 1

        # Add new tasks if prerequisites are met
        for key, prerequisites in task_prerequisites.items():
            if all(prereq in resource_manager.completed_tasks for prereq in prerequisites):
                if key not in available_tasks:
                    available_tasks.append(key)

    return build_order


def prerequisite_check(build_order):
    completed_tasks = set()
    for task_info in build_order:
        task_name = task_info["task"]
        if task_name in task_prerequisites:
            if not all(prereq in completed_tasks for prereq in task_prerequisites[task_name]):
                return False
        completed_tasks.add(task_name)
    return True


def fitness(build_order):
    resource_manager = ResourceManager()
    build_queue = sorted(build_order, key=lambda x: x["build_index"])

    for task_info in build_queue:
        task_name = task_info["task"]

        if task_name in task_prerequisites and not all(
                prereq in resource_manager.completed_tasks for prereq in task_prerequisites[task_name]
        ):
            return -1, {}

        resource_manager.build_task(task_name)

    economy_fitness = (
                              min(resource_manager.metal_rate / 40, 1.0) +
                              min(resource_manager.energy_rate / 600, 1.0)
                      ) / 2

    energy_penalty = max(
        (resource_manager.energy_rate - resource_manager.metal_rate * 6) / 100,
        0
    )

    # Adjust fitness based on whether goal rates are met
    if resource_manager.is_goal_met():
        economy_fitness += 0.2  # Bonus for achieving goal rates

    final_fitness = economy_fitness - energy_penalty

    extra_stats = {
        "energy_produced": resource_manager.energy_rate * resource_manager.time_elapsed,
        "metal_produced": resource_manager.metal_rate * resource_manager.time_elapsed,
        "total_time": resource_manager.time_elapsed,
    }

    return final_fitness, extra_stats


def get_fitness(build_order):
    return fitness(build_order)[0]


def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    return max(selected, key=get_fitness)


def crossover(parent1, parent2):
    if len(parent1) <= 1 or len(parent2) <= 1:
        return parent1, parent2  # If parents are too small, return them unchanged

    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    # Validate prerequisites for both children
    if not prerequisite_check(child1) or not prerequisite_check(child2):
        return parent1, parent2  # If invalid, return parents

    return child1, child2


def mutation(chromosome, mutation_rate=0.5):
    if len(chromosome) <= 1:
        return chromosome  # If the chromosome is too small, return unchanged

    # Apply mutation based on mutation rate
    if random.random() < mutation_rate:
        idx1 = random.randint(0, len(chromosome) - 1)
        idx2 = random.randint(0, len(chromosome) - 1)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    return chromosome


def genetic_algorithm(generations, population_size, max_tasks, tournament_size, mutation_rate):
    # Initialize the population
    population = [random_build_order(max_tasks) for _ in range(population_size)]

    # Find the best build order and its fitness
    def get_best_build_order(population):
        best_order = max(population, key=get_fitness)
        best_fitness, best_stats = fitness(best_order)
        return best_order, best_fitness, best_stats

    # Main loop for genetic algorithm
    for generation in range(generations):
        new_population = []

        best_build_order, best_fitness, best_stats = get_best_build_order(population)
        new_population.append(best_build_order)  # Keep the best one

        for _ in range((population_size - 1) // 2):
            parent1 = tournament_selection(population, k=tournament_size)
            parent2 = tournament_selection(population, k=tournament_size)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1, mutation_rate=mutation_rate))
            new_population.append(mutation(child2, mutation_rate=mutation_rate))

        population = new_population  # Update the population

        print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")

        if best_fitness > 0.9:  # Early exit condition
            break

    # Handle no valid build orders
    if best_stats is None:
        raise RuntimeError("No valid build order found")

    # Return the best build order and its statistics
    print("Optimal build order:", best_build_order)
    print("Energy produced:", best_stats["energy_produced"])
    print("Metal produced:", best_stats["metal_produced"])
    print("Total time:", best_stats["total_time"])

    return best_build_order


# Function to perform parameter tuning
def tune_parameters(runs, param_ranges, max_tasks):
    best_build_order = None
    best_fitness = -1
    best_params = {}

    # Randomize parameter selection order to reduce bias
    param_keys = list(param_ranges.keys())
    random.shuffle(param_keys)

    # Use a list to store fitness results for analysis
    results = []

    # Use a thread pool to run multiple tuning processes in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a generator for parameter combinations
        param_combinations = itertools.islice(itertools.cycle(product(
            param_ranges['population_size'],
            param_ranges['generations'],
            param_ranges['tournament_size'],
            param_ranges['mutation_rate']
        )), runs)

        # Run the genetic algorithm for each parameter combination
        futures = []
        for param_set in param_combinations:
            population_size, generations, tournament_size, mutation_rate = param_set
            future = executor.submit(
                genetic_algorithm,
                generations=generations,
                population_size=population_size,
                max_tasks=max_tasks,
                tournament_size=tournament_size,
                mutation_rate=mutation_rate
            )
            futures.append(future)

        # Process the results to find the best fitness
        for future in concurrent.futures.as_completed(futures):
            try:
                build_order = future.result()
                current_fitness, _ = fitness(build_order)
                results.append((build_order, current_fitness, param_set))
            except Exception as e:
                print("Error during genetic algorithm run:", e)

    # Find the best build order and parameters
    for build_order, current_fitness, param_set in results:
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_build_order = build_order
            best_params = {
                "population_size": param_set[0],
                "generations": param_set[1],
                "tournament_size": param_set[2],
                "mutation_rate": param_set[3],
            }

    return best_build_order, best_params, best_fitness


# Parameter ranges for tuning
parameter_ranges = {
    "population_size": list(range(50, 201, 50)),
    "generations": list(range(1000, 5001, 1000)),
    "tournament_size": [3, 5, 7],
    "mutation_rate": [0.2, 0.3, 0.4, 0.5],
}

RUN_GENETIC_ALGORITHM = True  # Set to True to run the genetic algorithm
RUN_PARAMETER_TUNING = False  # Set to True to run parameter tuning

if __name__ == "__main__":
    if RUN_GENETIC_ALGORITHM:
        optimal_build_order = genetic_algorithm(
            generations=2000,
            population_size=50,
            max_tasks=100,
            tournament_size=5,
            mutation_rate=0.5
        )
        # Loop through the list and print each task on a separate line
        print("Optimal build order:")
        for task in optimal_build_order:
            print(task)

    if RUN_PARAMETER_TUNING:
        # Define parameter ranges for tuning
        optimal_build_order, optimal_params, optimal_fitness = tune_parameters(
            runs=20,  # Number of random combinations to try
            param_ranges=parameter_ranges,
            max_tasks=100,
        )
        print("Optimal parameters:", optimal_params)
        print("Optimal fitness:", optimal_fitness)
        print("Optimal build order:", optimal_build_order)
