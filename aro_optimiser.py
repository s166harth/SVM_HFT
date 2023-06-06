import numpy as np
import random
class AROOptimizer:
    def __init__(self, model, population_size=50, max_iterations=100, alpha=0.5, beta=0.5, gamma=0.1):
        self.model = model
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def optimize(self, X_train, y_train):
        # Initialize the population of rabbits
        population = []
        for _ in range(self.population_size):
            rabbit = self.initialize_rabbit()
            population.append(rabbit)

        # Iterate through generations
        for iteration in range(self.max_iterations):
            # Evaluate fitness for each rabbit in the population
            fitness_values = []
            for rabbit in population:
                self.update_model_weights(rabbit)
                fitness = self.calculate_fitness(X_train, y_train)
                fitness_values.append(fitness)

            # Select the best rabbits for reproduction
            selected_rabbits = self.selection(population, fitness_values)

            # Create new rabbits through crossover and mutation
            new_population = self.reproduction(selected_rabbits)

            # Update the population
            population = new_population

        # Select the best rabbit as the optimized model
        best_rabbit = self.selection(population, fitness_values)[0]
        self.update_model_weights(best_rabbit)

    def initialize_rabbit(self):
        # Get the initial model weights
        initial_weights = self.model.get_weights()

        # Create a new rabbit with random perturbations
        rabbit = []
        for weight in initial_weights:
            perturbation = np.random.uniform(-1, 1, size=weight.shape)
            rabbit.append(weight + perturbation)

        return rabbit

    def update_model_weights(self, rabbit):
        # Update the model weights with the given rabbit
        self.model.set_weights(rabbit)

    def calculate_fitness(self, X_train, y_train):
        # Train the model and evaluate the fitness
        self.model.fit(X_train, y_train, epochs=1, batch_size=16, verbose=0)
        loss = self.model.evaluate(X_train, y_train, verbose=0)
        return 1 / (1 + loss)
    
    def selection(self, population, fitness_values):
    # Select the best rabbits based on fitness values
        sorted_population = [rabbit for _, rabbit in sorted(zip(fitness_values, population), reverse=True)]
        selected_rabbits = sorted_population[:int(self.alpha * len(population))]

        flattened_rabbits = np.concatenate(selected_rabbits, axis=None)

        return flattened_rabbits



    def reproduction(self, selected_rabbits):
        new_population = []

        while len(new_population) < self.population_size:
        # Perform crossover and mutation to create new rabbits
            parent_indices = random.sample(range(len(selected_rabbits)), k=2)
            parent1_index = parent_indices[0]
            parent2_index = parent_indices[1]

            parent1_weights = selected_rabbits[parent1_index]
            parent2_weights = selected_rabbits[parent2_index]

            child = self.crossover(parent1_weights, parent2_weights)
            mutated_child = self.mutation(child)
            new_population.append(mutated_child)

        return new_population


   

    def crossover(self, parent1, parent2):
      child = []
      for weight1, weight2 in zip(parent1, parent2):
            if weight1.shape != weight2.shape:
                # Reshape or resize the arrays to match their dimensions
                common_shape = (max(weight1.size, weight2.size),)
                weight1 = np.resize(weight1, common_shape)
                weight2 = np.resize(weight2, common_shape)
                # Perform crossover operation on the aligned arrays
                # Add the resulting child weights to the child list
            child_weights = (weight1 + weight2) / 2.0
            child.append(child_weights)
      return child
 
 


    def mutation(self, child):
        # Perform random perturbations to the child
        mutated_child = []
        for weight in child:
            perturbation = np.random.normal(loc=0, scale=self.gamma, size=weight.shape)
            mutated_weight = weight + perturbation
            mutated_child.append(mutated_weight)

        return mutated_child
