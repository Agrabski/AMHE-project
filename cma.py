import tensorflow as tf;
from cma import CMA;
from tensorflow import Tensor;
import numpy as np;
from sklearn.neighbors import NearestNeighbors;
import statistics

def weighted_neighbour_avg_discarding_cma(fitness_function , initial_solution, initial_step_size, k, v):
	class real_fitness_function:
		def __init__(self):
			self._fitness = fitness_function;
			self._previous_generation = None;
			self._avg_previous_variance = 0;

		def __call__(self, x: Tensor) -> Tensor:
			if(self._previous_generation == None):
				self._previous_generation = {};
				for elem in x:
					self._previous_generation[elem.numpy()] = self._fitness(x);
			else:
				new_variance = 0;
				new_generation = {};
				all_previous_specimen = self._previous_generation.keys();
				for elem in x:
					set = np.array(elem);
					set.__add__(all_previous_specimen);
					nbrs = NearestNeighbors(n_neighbors=k).fit(set);
					distances, neighbours = nbrs.kneighbors();
					neighbour_indicies = [a[1] for a in neighbours if a[0] == 0];
					evaluations = [self._previous_generation[set[i]] for i in neighbours];
					variance = statistics.variance(evaluations);

					if variance > self._avg_previous_variance * v:
						new_generation[elem.numpy()] = self._fitness(elem);
					else:
						weighted_evals = [evaluations[i] * distances[neighbours.index([0, i])][1] for i in neighbour_indicies];

						new_generation[elem.numpy()] = sum(weighted_evals) / sum([distances[neighbours.index([0, i])][1] for i in neighbour_indicies])

					new_variance += variance;

				self._avg_previous_variance = new_variance / x._shape_as_list()[0];
				self._previous_generation = new_generation;
				return tf.convert_to_tensor(new_generation.values());





	return CMA(
		initial_solution,
		initial_step_size,
		real_fitness_function);



def low_variance_neighborhood_average_cma(fitness_function, initial_solution, initial_step_size):
	def real_fitness_function(x):
		return fitness_function(3);
	
	return CMA(
		initial_solution,
		initial_step_size,
		real_fitness_function);
	
