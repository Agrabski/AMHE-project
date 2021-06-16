from cma2_electric_boogaloo import weighted_neighbour_avg_discarding_cma, low_variance_neighborhood_average_cma
import cec2017.functions as cec
#zeby to dzialalo trzeba pobrac i zainstalować to https://github.com/tilleyd/cec2017-py
#być może jeszcze skopiowac ten plik -> info w errorze co brakuje, plik bedzie w tym repo
#zainstalowac albo w env1 tym automatycznym jak wiesz jak albo miec py 3.7 na kompie i w nim -> py bleble install
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


functions = [(2,cec.f2),(3, cec.f3),(6, cec.f6)]
dimentions = [2,10,50]

runs = 10
population = 100
searches = 100
stepSize = 1.0

kVals = [25]
vVals = [0.6, 0.8, 1, 1.2, 1.4]
bestV = 1
bestK = 7

testK = False
testDiscard = False

if testK:
	vVals = [bestV]
else:
	kVals = [bestK]

if testDiscard:
	testFun = weighted_neighbour_avg_discarding_cma
else:
	testFun = low_variance_neighborhood_average_cma

results = []
for kVal in kVals:
	for vVal in vVals:
		fResult = []
		for i, f in functions:
			dResult = []
			for d in dimentions:
				sResult = SettingResult(i, d, [Param("k", kVal), Param("v", vVal)])
				cmaResult = SettingResult(i, d, [])
				for run in range(runs):
					#potrzebny jest ten logger albo cus co pozwoli nam podgladac best w danej iteracji CMA
					#aktualnie nasze czasami ogarnia ze [100, -100] jest najlepsze ze score=3000 zato CMA zwykle pełza wokół 200????!!! Co jest chyba min w tym rejonie O.o
					randStart = np.random.uniform(-100, 100, size=d)
					cmaes = testFun(
						f,
						initial_solution=randStart ,
						initial_step_size=stepSize,
						populationSize = population,
						enforceBounds = np.repeat(np.array([[-100,100]]), d, axis= 0),
						k=kVal,
						v=vVal,
					)		

					start_time = time.time()
					with tf.device('/GPU:0'):
						try:
							solution, fitness = cmaes.search(searches)
							t = time.time() - start_time
							sResult.add(Result(solution, f(solution), t))
						except:
							print("our fail for: " + str(i) + " | " + str(d))

					def fun(x):
						return tf.convert_to_tensor([-f(point) for point in x.numpy()])

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
					with tf.device('/GPU:0'):
						try:
							solution, fitness = cmaes.search(searches)
							t = time.time() - start_time
							cmaResult.add(Result(solution, -fitness, t))
						except:
							print("our fail for: " + str(i) + " | " + str(d))
					print("run no." + str(run) + " DONE")


				fn = sResult.fileName()
				pickle.dump(sResult, open("results/" + ("discard/" if testDiscard else "weight/") + str(fn) + ".pickle" , "wb" ) )
				dResult.append((sResult, cmaResult))


			
			print("function no." + str(i) + " DONE")
			fResult.append(dResult)
	print("kVal " + str(kVal) + "vVal " + str(vVal) + " DONE")
	results.append(fResult)

pickle.dump(results, open( "results/" + ("discard/" if testDiscard else "weight/") + "all.pickle" , "wb" ) )

for paramI, paramR in enumerate(results):
	print("k=" + str(kVals[paramI]) if testK else "v=" + str(vVals[paramI]))
	paramFitness = 0
	paramTime = 0
	for fi, fr in enumerate(paramR):
		print("\tfunc i=" + str(functions[fi][0]))
		fFitness = 0
		fTime = 0
		count = 0
		for id, dr in enumerate(fr):
			try:
				print("\t\td=" + str(dimentions[id]))
				our, cmaes = dr
				fitness1, t1 = our.median()
				fitness2, t2 = cmaes.median()
				count +=1
				print("\t\t\tf=" + str(our.best().fitness) +  "  t=" + str(our.best().time)) #+ "  point=" + str(our.best().solution))
				print("\t\t\tf=" + str(cmaes.best().fitness) +  "  t=" + str(cmaes.best().fitness))# + "  point=" + str(cmaes.best().solution))
				print("\t\tFor dimention median fitness=" + str(fitness1/fitness2) + " and median time=" + str(t1/t2))
				fFitness += fitness1/fitness2
				fTime += t1/t2
			except:
				pass
		
		try:
			print("\t\tFor function, all dimention median fitness=" + str(fFitness/len(fr)) + " and median time=" + str(fTime/len(fr)))
			paramFitness += fFitness/len(fr)
			paramTime += fTime/len(fr)
		except:
			pass
	try:
		print("\tFor param, all function and dimention median fitness=" + str((paramFitness/len(paramR))) + " and median time=" + str(paramTime/len(paramR)))
	except:
		pass
	

#bo czasami okno sie samo zamyka :|
x=0

