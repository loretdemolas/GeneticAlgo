# main.py
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from parameter_tuning import ParameterTuning


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


RUN_GENETIC_ALGORITHM = True  # Set to True to run the genetic algorithm

if __name__ == "__main__":
    if RUN_GENETIC_ALGORITHM:
        # Run the genetic algorithm
        ga = GeneticAlgorithm(
            generations=600,
            population_size=300,
            tournament_size=3,
            initial_mutation_rate=0.7,
            increase_factor=1.05,
            decrease_factor=.95,
            threshold=0.05,
        )
        best_order, best_fitness, best_stats = ga.run()

        # Print the best build order
        print("Best Build Order:")
        for task in best_order:
            task_name = task["task"]
            build_time = task.get("build_time", "N/A")
            current_metal = task.get("current_metal")
            current_energy = task.get("current_energy")
            print(
                f"- {task_name}, Build Time: {build_time:.2f} seconds, M: {current_metal}, E: {current_energy}"
            )

        # Print the best stats
        print("\nBest Stats:")
        for key, value in best_stats.items():
            print(f"{key}: {value}")

        # Plot the best fitness over generations
        plt.plot(range(len(ga.fitness_per_generation)), ga.fitness_per_generation, "bo-", markersize=5)
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
        plt.plot(range(len(tuner.results)), [r["best_fitness"] for r in tuner.results], "bo-", markersize=5)
        plt.title("Best Fitness for Different Configurations")
        plt.xlabel("Configuration Index")
        plt.ylabel("Best Fitness")
        plt.show()
