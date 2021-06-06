from cma import CMA
from enum import Enum
class CmaVariant(Enum):
	low_variance_neighborhood_average = 1,
	weighted_neighbour_avg_discarding = 2,

def weighted_neighbour_avg_discarding_cma(fitness_function, variant, initial_solution=[1.5, -0.4], initial_step_size=1.0):
	def real_fitness_function(x):
		return fitness_function(3);
	
	return CMA(
		initial_solution,
		initial_step_size,
		real_fitness_function);
def low_variance_neighborhood_average_cma(fitness_function, variant, initial_solution=[1.5, -0.4], initial_step_size=1.0):
	def real_fitness_function(x):
		return fitness_function(3);
	
	return CMA(
		initial_solution,
		initial_step_size,
		real_fitness_function);
	
