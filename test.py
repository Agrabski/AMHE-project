from cma2_electric_boogaloo import weighted_neighbour_avg_discarding_cma, low_variance_neighborhood_average_cma

def fitness_fn(x):
	return (
		(4 - 2.1 * x[0]**2 + x[0]**4 / 3) * x[0]**2 +
		x[0] * x[1] +
		(-4 + 4 * x[1]**2) * x[1]**2
	)


cma = weighted_neighbour_avg_discarding_cma(
	fitness_fn,
	initial_solution=[1.5, -0.4],
	initial_step_size=1.0,
	k=10,
	v=1,
)

cma2 = low_variance_neighborhood_average_cma(
	fitness_fn,
	initial_solution=[1.5, -0.4],
	initial_step_size=1.0,
	k=10,
	s=1,
	population_size=100
)
best_solution, best_fitness = cma.search()
best_solution2, best_fitness2 = cma2.search()
print(best_solution)
print(best_fitness)
print(best_solution2)
print(best_fitness2)