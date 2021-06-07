from cma2_electric_boogaloo import weighted_neighbour_avg_discarding_cma

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
best_solution, best_fitness = cma.search()
print(best_solution)
print(best_fitness)