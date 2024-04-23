import random

# Task definitions with costs and build times
tasks = {
    "Build Factory": {"metal_cost": 620, "energy_cost": 1300, "base_build_time": 65},
    "Build Builder": {"metal_cost": 120, "energy_cost": 1750, "base_build_time": 36},
    "Build Metal Extractor": {"metal_cost": 50, "energy_cost": 500, "base_build_time": 19},
    "Build Solar Collector": {"metal_cost": 150, "energy_cost": 0, "base_build_time": 28},
    "Build Energy Converter": {"metal_cost": 1, "energy_cost": 1250, "base_build_time": 27},
    "Wait 1 Minute": {"metal_cost": 0, "energy_cost": 0, "base_build_time": 60},
}

# Prerequisites for tasks
task_prerequisites = {
    "Build Builder": ["Build Factory"],
}

# Check if a build order is valid given the prerequisites
def prerequisite_check(build_order):
    completed_tasks = set()
    for task_info in build_order:
        task_name = task_info["task"]
        if task_name in task_prerequisites:
            if not all(prereq in completed_tasks for prereq in task_prerequisites[task_name]):
                return False
        completed_tasks.add(task_name)
    return True

# Generate a random build order with resource constraints and prerequisites check
import random

# Updated random build order with separate build index and time tracking
def random_build_order(max_tasks, initial_metal=1000, initial_energy=1000):
    # Resource tracking and initial settings
    build_order = []
    build_index = 0
    current_metal = initial_metal
    current_energy = initial_energy
    build_power = 400
    metal_rate = 0
    energy_rate = 20  # Initial energy rate
    time_elapsed = 0

    # Define the ideal energy-to-metal ratio
    energy_to_metal_ratio = 70  # Energy required for 1 metal

    available_tasks = ["Build Metal Extractor", "Build Solar Collector"]

    for _ in range(max_tasks):
        valid_tasks = [task for task in available_tasks if
                       current_metal >= tasks[task]["metal_cost"] and
                       current_energy >= tasks[task]["energy_cost"]]

        if not valid_tasks:
            break  # No valid tasks can be built, stop generating

        # Adjust task probabilities to maintain resource balance
        task_probabilities = {
            "Build Metal Extractor": 1.0 if energy_rate >= metal_rate * energy_to_metal_ratio else 0.5,
            "Build Solar Collector": 1.0 if metal_rate >= energy_rate / energy_to_metal_ratio else 0.5,
            "Build Energy Converter": 1.0 if energy_rate > metal_rate * energy_to_metal_ratio else 0.5,
        }

        # Filter valid tasks with updated probabilities
        valid_tasks = [task for task in valid_tasks if task_probabilities[task] > 0]

        # If no valid tasks meet the constraints, stop
        if not valid_tasks:
            break

        # Choose a task with weighted probabilities
        task = random.choices(valid_tasks, weights=[task_probabilities[t] for t in valid_tasks])[0]

        # Add task to build order
        build_order.append({"task": task, "build_index": build_index})

        # Calculate build time and update resource counts
        build_time = tasks[task]["base_build_time"] / (build_power / 100)
        current_metal -= tasks[task]["metal_cost"]
        current_energy -= tasks[task]["energy_cost"]

        # Update metal and energy production rates
        if task == "Build Metal Extractor":
            metal_rate += 1.8  # Increase metal production
        elif task == "Build Solar Collector":
            energy_rate += 20  # Increase energy production
        elif task == "Build Energy Converter":
            energy_rate -= 70
            metal_rate += 1 # Decrease energy, increase metal production
        elif task == "Build Builder":
            build_power += 100  # Increase build power

        # Update time and build order index
        time_elapsed += build_time
        build_index += 1

        # Add tasks based on prerequisites
        for task_name, prerequisites in task_prerequisites.items():
            if all(prereq in [bo["task"] for bo in build_order] for prereq in prerequisites):
                available_tasks.append(task_name)

    return build_order


def fitness(build_order):
    # Initialize resources and build queue
    current_metal = 1000
    current_energy = 1000
    build_power = 400
    metal_rate = 0
    energy_rate = 20  # Initial energy rate
    build_queue = sorted(build_order, key=lambda x: x["build_index"])
    time_elapsed = 0
    energy_to_metal_ratio = 70

    metal_produced = 0
    energy_produced = 0
    valid_build = True
    completed_tasks = set()

    # Process the build order
    for task_info in build_queue:
        task_name = task_info["task"]

        if task_name in task_prerequisites and not all(prereq in completed_tasks for prereq in task_prerequisites[task_name]):
            valid_build = False
            break  # Prerequisite check failed

        task_details = tasks[task_name]
        build_time = task_details["base_build_time"] / (build_power / 100)

        # Ensure enough resources to build the task
        if current_metal >= task_details["metal_cost"] and current_energy >= task_details["energy_cost"]:
            current_metal -= task_details["metal_cost"]
            current_energy -= task_details["energy_cost"]

            # Update resource rates based on the task being built
            if task_name == "Build Metal Extractor":
                metal_rate += 1.8  # Increase metal rate
            elif task_name == "Build Solar Collector":
                energy_rate += 20  # Increase energy rate
            elif task_name == "Build Energy Converter":
                energy_rate -= 70  # Energy to metal conversion
                metal_rate += 1
            elif task_name == "Build Builder":
                build_power += 100  # Increase build power

            time_interval = build_time
            metal_produced += time_interval * metal_rate
            energy_produced += time_interval * energy_rate
            time_elapsed += build_time  # Update elapsed time

            completed_tasks.add(task_name)

        else:
            valid_build = False
            break  # Insufficient resources

    if not valid_build:
        return -1, {}  # Invalid build order

    # Fitness calculation with energy-to-metal ratio balance
    economy_fitness = (min(metal_rate / 80, 1.0) + min(energy_rate / 100, 1.0)) / 2
    energy_penalty = max((energy_rate - metal_rate * energy_to_metal_ratio) / 100, 0)  # Penalize excess energy
    final_fitness = economy_fitness - energy_penalty

    extra_stats = {
        "energy_produced": energy_produced,
        "metal_produced": metal_produced,
        "total_time": time_elapsed,
    }

    return final_fitness, extra_stats


# Selection, crossover, mutation, and genetic algorithm
def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    return max(selected, key=lambda x: fitness(x)[0])

def crossover(parent1, parent2):
    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    if not prerequisite_check(child1) or not prerequisite_check(child2):
        return parent1, parent2  # If invalid, return parents

    return child1, child2

def mutation(chromosome, mutation_rate=0.5):
    if random.random() < mutation_rate:
        idx1 = random.randint(0, len(chromosome) - 1)
        idx2 = random.randint(0, len(chromosome) - 1)
        # Swap dictionaries at idx1 and idx2
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome


def genetic_algorithm(generations, population_size, max_tasks):
    population = [random_build_order(max_tasks) for _ in range(population_size)]

    best_build_order = None
    best_stats = None
    best_fitness = -1

    for generation in range(generations):
        new_population = []

        best_build_order = max(population, key=lambda x: fitness(x)[0])
        best_fitness, best_stats = fitness(best_build_order)
        new_population.append(best_build_order)

        for _ in range((population_size - 1) // 2):
            parent1 = tournament_selection(population, k=5)
            parent2 = tournament_selection(population, k=5)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1, mutation_rate=0.5))
            new_population.append(mutation(child2, mutation_rate=0.5))

        population = new_population

        print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")

        if best_fitness > 0.9:
            break

    if best_stats is None:
        raise RuntimeError("No valid build order found")

    print("Optimal build order:", best_build_order)
    print("Energy produced:", best_stats["energy_produced"])
    print("Metal produced:", best_stats["metal_produced"])
    print("Total time:", best_stats["total_time"])

    return best_build_order

# Example usage
if __name__ == "__main__":
    optimal_build_order = genetic_algorithm(
        generations=5000,
        population_size=20,
        max_tasks=100,
    )
