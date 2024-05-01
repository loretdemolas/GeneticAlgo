# parameter_tuning.py
from genetic_algorithm import GeneticAlgorithm


class ParameterTuning:
    def __init__(self, generations, population_sizes, tournament_sizes, mutation_rates):
        self.generations = generations
        self.population_sizes = population_sizes
        self.tournament_sizes = tournament_sizes
        self.mutation_rates = mutation_rates

        # Store tuning results
        self.results = []

    def tune_parameters(self):
        for population_size in self.population_sizes:
            for tournament_size in self.tournament_sizes:
                for mutation_rate in self.mutation_rates:
                    # Run the GA with the current configuration
                    ga = GeneticAlgorithm(
                        generations=self.generations,
                        population_size=population_size,
                        tournament_size=tournament_size,
                        initial_mutation_rate=mutation_rate,
                        increase_factor=1.05,
                        decrease_factor=.95,
                        threshold=0.05,
                    )
                    best_order, best_fitness, best_stats = ga.run()

                    # Store results for analysis
                    self.results.append({
                        "population_size": population_size,
                        "tournament_size": tournament_size,
                        "mutation_rate": mutation_rate,
                        "best_fitness": best_fitness,
                        "best_order": best_order,
                        "best_stats": best_stats,
                    })

    def get_best_configuration(self):
        # Return the configuration with the highest fitness
        return max(self.results, key=lambda result: result["best_fitness"])
