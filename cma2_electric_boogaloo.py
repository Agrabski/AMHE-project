import tensorflow as tf;
from cma import CMA;
from tensorflow import Tensor;
import numpy as np;
from sklearn.neighbors import NearestNeighbors;
import statistics
import math

def low_variance_neighborhood_average_cma(fitness_function , initial_solution, initial_step_size, populationSize, enforceBounds, k, v, logger):
	class real_fitness_function:
		def __init__(self):
			self._fitness = fitness_function;
			self._previous_generation = None;
			self._avg_previous_variance = 0;

		def __call__(self, x: Tensor) -> Tensor:
			if(self._previous_generation == None):
				self._previous_generation = [];
				for elem in x:
					z = elem.numpy()
					self._previous_generation.append((elem, -self._fitness(elem.numpy())));
				return tf.convert_to_tensor([p[1] for p in self._previous_generation]);

			else:
				new_variance = 0;
				new_generation = [];
				set = [p[0].numpy() for p in self._previous_generation]
				setFits = [p[1] for p in self._previous_generation]
				for elem in x:
					nbrs = NearestNeighbors(n_neighbors=k).fit(set);
					distances, neighbours = nbrs.kneighbors([elem.numpy()]);
					neighbour_indicies = list(neighbours[0]);
					evaluations = [setFits[i] for i in neighbour_indicies];
					variance = statistics.variance(evaluations);

					if 0 in distances[0]:
						fit = setFits[neighbour_indicies[list(distances[0]).index(0)]]
						new_generation.append((elem, fit))
					elif variance > self._avg_previous_variance * v:
						fit = self._fitness(elem.numpy())
						new_generation.append((elem, -fit));
						temp = list(set)
						temp.append(elem.numpy());
						set = np.array(temp)
						setFits.append(fit)
					else:
						weighted_evals = [setFits[i2] * distances[0][i1] for i1, i2 in enumerate(neighbour_indicies)];
						fit = sum(weighted_evals) / sum([distances[0][i] for i in range(len(distances[0]))])
						new_generation.append((elem, fit))

					new_variance += variance;

				self._avg_previous_variance = new_variance / x._shape_as_list()[0];
				self._previous_generation = new_generation
				j = [p[1] for p in new_generation]
				logger(min([p[1] for p in new_generation]))
				return tf.convert_to_tensor([p[1] for p in new_generation]);





	return CMA(
		initial_solution,
		initial_step_size,
		real_fitness_function(),
		population_size = populationSize,
		enforce_bounds = enforceBounds);



def weighted_neighbour_avg_discarding_cma(fitness_function, initial_solution, initial_step_size, enforceBounds, k, v, population_size):
	class real_fitness_function:
		def __init__(self):
			self._fitness = fitness_function;
			self._previous_generation = None;

		def __call__(self, x: Tensor) -> Tensor:
			if(self._previous_generation == None):
				self._previous_generation = {};
				for elem in x:
					self._previous_generation[tuple(elem.numpy())] = self._fitness(elem).numpy();
				return tf.convert_to_tensor(list(self._previous_generation.values()));

			else:
				previous_median = statistics.median(list(self._previous_generation.values()))
				new_generation = {};
				all_previous_specimen = list(self._previous_generation.keys());
				for elem in x:
					set = [elem.numpy()];
					for specimen in all_previous_specimen:
						set.append(specimen);
					nbrs = NearestNeighbors(n_neighbors=k).fit(set);
					distances, neighbours = nbrs.kneighbors();
					neighbour_indicies = neighbours[0];
					neighbour_indicies = [index for index in neighbour_indicies if tuple(elem.numpy()) != set[index]]

					weighted_evals = [self._previous_generation[tuple(list(set[index]))] * distances[0][i - 1] for i, index in enumerate(neighbour_indicies)];
					weighted_evals = [e for e in weighted_evals if np.isfinite(e)]
					weighted_avg_eval = sum(weighted_evals) / sum([distances[0][i] for i, index in enumerate(neighbour_indicies)]);
					if weighted_avg_eval > previous_median * v:
						new_generation[tuple(elem.numpy())] = self._fitness(elem).numpy();
					else:
						new_generation[tuple(elem.numpy())] = float('-inf');


				result = tf.convert_to_tensor(list(new_generation.values()));
				self._previous_generation = new_generation;
				return result;
	
	return CMA(
		initial_solution,
		initial_step_size,
		real_fitness_function(),
		population_size=population_size,
		enforce_bounds = enforceBounds);
	
