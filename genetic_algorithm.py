# genetic_algorithm.py
import random
from build_order_fitness import BuildOrderFitness
from build_order_manager import BuildOrderManager
from resource_manager import ResourceManager


def crossover(parent1, parent2):
    if len(parent1) <= 1 or len(parent2) <= 1:
        return parent1, parent2

    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    if not BuildOrderFitness(child1).prerequisite_check() or not BuildOrderFitness(child2).prerequisite_check():
        return parent1, parent2

    return child1, child2


def random_build_order():
    resource_manager = ResourceManager()
    build_order_manager = BuildOrderManager(resource_manager)
    return build_order_manager.create_build_order()


def count_task_types(build_order):
    # Create a dictionary to count each task type
    task_count = {}
    for task_info in build_order:
        task_name = task_info["task"]
        if task_name in task_count:
            task_count[task_name] += 1
        else:
            task_count[task_name] = 1
    return task_count


class GeneticAlgorithm:
    def __init__(self, generations, population_size, tournament_size, initial_mutation_rate, increase_factor,
                 decrease_factor, threshold):
        self.generations = generations
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.initial_mutation_rate = initial_mutation_rate
        self.mutation_rate = initial_mutation_rate
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.threshold = threshold
        self.population = []
        self.fitness_per_generation = []
        self.stagnation_limit = 20  # Number of generations with no improvement before increasing mutation rate
        self.stagnation_counter = 0

        # Populate the initial population
        for _ in range(self.population_size):
            self.population.append(random_build_order())

    def adjust_mutation_rate(self, previous_best, current_best):
        if current_best > previous_best + self.threshold:
            self.mutation_rate *= self.decrease_factor  # Decrease mutation rate if fitness improves
            self.stagnation_counter = 0
        else:
            self.mutation_rate *= self.increase_factor  # Increase mutation rate if fitness stagnates
            self.stagnation_counter += 1

        # Check stagnation and reintroduce new individuals if needed
        if self.stagnation_counter >= self.stagnation_limit:
            self.mutation_rate *= 1.5  # Rapidly increase mutation rate
            for _ in range(5):
                self.add_random_individual()
            self.stagnation_counter = 0  # Reset stagnation counter
            self.manage_population()

        self.mutation_rate = min(max(self.mutation_rate, 0.1), 1.0)

    def add_random_individual(self):
        new_order = random_build_order()
        self.population.append(new_order)

    def manage_population(self):
        if len(self.population) > self.population_size:
            self.population = sorted(
                self.population, key=lambda bo: BuildOrderFitness(bo).get_fitness(), reverse=True
            )[:self.population_size]

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
        previous_best_fitness = float('-inf')

        for generation in range(self.generations):
            new_population = []

            # Find the best order and add to the new population
            best_order, best_fitness, best_stats = self.get_best_build_order()
            self.fitness_per_generation.append(best_fitness)
            new_population.append(best_order)

            self.adjust_mutation_rate(previous_best_fitness, best_fitness)
            previous_best_fitness = best_fitness

            # Add task type counts to the best stats
            task_counts = count_task_types(best_order)
            best_stats.update(task_counts)

            # Create the rest of the population through crossover and mutation
            for _ in range((self.population_size - 1) // 2):
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))

            self.population = new_population

            print(
                f"Generation {generation}: Best Fitness = {best_fitness:.2f}\n"
                f"\tStagnation Counter = {self.stagnation_counter}/{self.stagnation_limit}\n"
                f"\tMutation Rate = {self.mutation_rate:.2f}"
            )

        return best_order, best_fitness, best_stats
