from cma2_electric_boogaloo import weighted_neighbour_avg_discarding_cma, low_variance_neighborhood_average_cma
import cec2017.functions as cec
import numpy as np
from cec2017.functions import all_functions
import time
import pickle
from cma import CMA
import tensorflow as tf;
import statistics


class Param:
	def __init__(self, name, value):
		self.name = name
		self.value = value

	def fileName(self):
		return "param{" + str(self.name) + "}=" + str(self.value)

class Result:
	def __init__(self, solution, fitness, time):
		self.solution = solution
		self.fitness = fitness
		self.time = time

class SettingResult:
	def __init__(self, function: int, dim: int, params):
		self.function = function
		self.dim = dim
		self.params = params
		self.results = []

	def fileName(self):
		s = "-"
		return "fun" + str(self.function) + "-dim" + str(self.dim) + "-" + str(s.join([x.fileName() for x in self.params]))

	def add(self, result):
		self.results.append(result)

	def best(self):
		temp = [x.fitness for x in self.results]
		index = temp.index(max(temp))
		return self.results[index]

	def worst(self):
		temp = [x.fitess for x in self.results]
		index = temp.index(min(temp))
		return self.results[index]

	def average(self):
		return sum([x.fitness for x in self.results])/len(self.results), sum([x.time for x in self.results])/len(self.results)

	def median(self):
		return statistics.median([x.fitness for x in self.results]), statistics.median([x.time for x in self.results])

'''
wtf is this... other values sometimes work
a = np.array([ 63.85945269, -14.87061561])
x = cec.f18(a)
'''

functions = [(2,cec.f2)]
dimentions = [2]
kVals = [5]
runs = 2
population = 10
searches = 10
stepSize = 1.0


kResult = []

vVal = 1

for kVal in kVals:
	fResult = []
	for i, f in functions:
		dResult = []
		for d in dimentions:
			sResult = SettingResult(i, d, [Param("k", kVal), Param("v", vVal)])
			cmaResult = SettingResult(i, d, [])
			for run in range(runs):
				#potrzebny jest ten logger albo cus co pozwoli nam podgladac best w danej iteracji CMA
				#aktualnie nasze czasami ogarnia ze [100, -100] jest najlepsze ze score=3000 zato CMA zwykle pełza wokół 200????!!! Co jest chyba min w tym rejonie O.o
				randStart = [0,0] #np.random.uniform(-100, 100, size=d)
				cmaes = low_variance_neighborhood_average_cma(
					f,
					initial_solution=randStart ,
					initial_step_size=stepSize,
					populationSize = population,
					enforceBounds = np.repeat(np.array([[-100,100]]), d, axis= 0),
					k=kVal,
					v=vVal,
				)		

				start_time = time.time()
				solution, fitness = cmaes.search(searches)
				t = time.time() - start_time
				sResult.add(Result(solution, f(solution), t))

				def fun(x):
					return tf.convert_to_tensor([f(point) for point in x.numpy()])

				#to udowadnia ze fun() zwraca wyniki poprawnie ->(ok. maks)=1>2>3>4 
				v = fun(tf.convert_to_tensor([[99,-99],[75,-75],[50,-50],[0,0]]))

				cmaes = CMA(
					randStart,
					stepSize,
					fun,
					population_size = population,
					enforce_bounds=np.repeat(np.array([[-100,100]]), d, axis= 0)
				)
				start_time = time.time()
				solution, fitness = cmaes.search(searches)
				t = time.time() - start_time
				cmaResult.add(Result(solution, fitness, t))
				#print("run no." + str(run) + " DONE")


			fn = sResult.fileName()
			pickle.dump(sResult, open( "results/discardingParam/" + str(fn) + ".pickle" , "wb" ) )
			dResult.append((sResult, cmaResult))


			
		#print("function no." + str(i) + " DONE")
		fResult.append(dResult)
	#print("kVal " + str(kVal) + " DONE")
	kResult.append(fResult)

pickle.dump(kResult, open( "results/discardingParam/all.pickle" , "wb" ) )

for ki, kr in enumerate(kResult):
	print("k=" + str(kVals[ki]))
	kFitness = 0
	kTime = 0
	for fi, fr in enumerate(kr):
		print("\tfunc i=" + str(functions[fi][0]))
		fFitness = 0
		fTime = 0
		count = 0
		for id, dr in enumerate(fr):
			print("\t\td=" + str(dimentions[id]))
			our, cmaes = dr
			fitness1, t1 = our.median()
			fitness2, t2 = cmaes.median()
			count +=1
			print("\t\t\tf=" + str(our.best().fitness) +  "  t=" + str(our.best().time) + "  point=" + str(our.best().solution))
			print("\t\t\tf=" + str(cmaes.best().fitness) +  "  t=" + str(cmaes.best().fitness) + "  point=" + str(cmaes.best().solution))
			fFitness += fitness1/fitness2
			fTime += t1/t2
			
		print("\t\tFor function, all dimention median fitness=" + str(fFitness/len(fr)) + " and median time=" + str(fTime/len(fr)))
		kFitness += fFitness/len(fr)
		kTime += fTime/len(fr)
	print("\tFor k, all function and dimention median fitness=" + str((kFitness/len(kr))) + " and median time=" + str(kTime/len(kr)))

#bo czasami okno sie samo zamyka :|
x=0

