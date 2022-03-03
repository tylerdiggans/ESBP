import random, argparse, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint #, solve_ivp
from scipy.spatial.distance import pdist
from copy import deepcopy
from MSFTools import MSF, PlotMSF
#############################################################################
###   This is used to simulate synchronization of oscillators  				#
###		on a network: There are 4 main programs								#	
###		Simulate_Synch - For prescribed adjacency in parallel runs 			#
###		Simulate_ICs   - Find a number of ICs that achieve synch (XOs)		#
### Find_C which calls:														#
###		MSF - Master Stability Function analysis to find coupling strength 	#
###		MSF_Liapunov   - Estimate the Largest Liapunov Exponent (LLE) 	  	#
#############################################################################

def get_input():
	parser = argparse.ArgumentParser()	
	#------------------Model Parameters----------------------------
	parser.add_argument('--system', default=None, 
						help='Identical chaotic attractor systems (defaults to Rossler)')
	parser.add_argument('--params', default='0.2, 0.2, 9.0', 
						help='parameters for basic chaotic attractor (defaults to Rossler)')
	parser.add_argument('--H', default='x', 
						help='coupling matrix defaults to x variable (use x,y,z, or 3x3 matrix)')
	parser.add_argument('--C', default=None, 
						help='coupling strength \
						An interval of possible values is found using MSF if unknown')
	parser.add_argument('--C_max', default=None, 
						help='limit on coupling strength?')
	parser.add_argument('--T_plot', default=100, 
						help='Time for plotting')
	parser.add_argument('--matrix', default=None, 
						help='Connectivity matrix L, $L_{rw}$, or $\mathcal{L}$')
	parser.add_argument('--Adj', default=None, 
						help='txt filename containing an adjacency matrix \
								saved using pandas.to_csv() so need to use \
								genfromtxt(skip_header=1, delimiter=","')
	parser.add_argument('--X0s', default=None, 
						help='txt filename containing IC that converge if none given, \
						then new ones are generated by Simulate_ICs and saved')
	parser.add_argument('--synchtol', default=1E-6, 
						help='tolerance to assume synchronization')
	parser.add_argument('--parallel', default=None, 
						help='average time to synch results over a number of parallel runs')
	parser.add_argument('--cores', default=None, 
						help='Howmany cores to use for pool')
	#------------------Plotting Parameters----------------------------
	parser.add_argument('--plotting', default=None, 
						help='Used to suppress plotting functionality')
	parser.add_argument('--saving', default=None, 
						help='Choose to save off either "ICs" or "Synch" or both sep by comma')
	args = parser.parse_args()
	return args         

def process_input(args):
	# Define coupling matrix by 'x' 'y' or 'z' or a 3x3 matrix
	H = np.zeros((3,3))       # as in Pecora/Carrol
	if args.H=='x':
		H[0,0] = 1
	elif args.H=='y':
		H[1,1] = 1
	elif args.H=='z':
		H[2,2] = 1
	else:
		H = np.array(args.H)
	# Import the chaotic attractor being used
	if not args.system:
		# Default to Rossler oscillators coupled through x
		from Attractor_Functions import rossler, rossler_net_ll, DF_rossler
		Func = rossler
		Func_net = rossler_net_ll
		DFunc = DF_rossler
	elif args.system == 'Lorenz':
		params = (10.,28.,8./3.)
		from Attractor_Functions import lorenz, lorenz_net_ll, DF_lorenz
		Func = lorenz
		Func_net = lorenz_net_ll
		DFunc = DF_lorenz
	elif args.system == 'Chens':
		from Attractor_Functions import chens, chens_net_ll, DF_chens
		Func = chens
		Func_net = chens_net_ll
		DFunc = DF_chens
	elif args.system == 'Harmonic':
		from Attractor_Functions import harm, harm_net_ll, DF_harm
		Func = harm
		Func_net = harm_net_ll
		DFunc = DF_harm
		H = H[:2,:2]
	elif args.system == 'Simple':
		from Attractor_Functions import simple, simple_net_ll, DF_simple
		Func = simple
		Func_net = simple_net_ll
		DFunc = DF_simple
		H = H[0,0]
	# Set up the parameters for the attractor
	if args.system=='Lorenz':
		params = (10.,28.,8./3.)
	else:
		params = tuple([float(eval(x)) for x in args.params.split(',')]) 

	##########################################################

	######################################
	# An adjacency matrix must be supplied
	######################################
	A = np.genfromtxt(args.Adj, skip_header=1, delimiter=',')
	G_name = args.Adj.split('/')[-1].split('.')[0]
	# option of supplying initial conditions that are known 
	#     to converge to save time on the generation of these
	#     If you ahve run it once, these can be saved and reused 
	if args.X0s:
		X0s = np.genfromtxt(args.X0s, skip_header=1, delimiter=',')
	else:
		X0s = None   # initial conditions will be obtained and saved
	if args.parallel:
		# If utilizing parallel threads to average over many runs
		nn = int(args.parallel)
	else:
		# Just do one run
		nn=1

	if args.cores:
		# option of assigning a number of cores to utilize
		cores = int(args.cores)
	else:
		cores = cpu_count()

	if not args.C_max:
		# If a maximum global coupling strength is imposed 
		# plays the role of sigma in the literature
		C_max = None
	else:
		C_max = int(args.C_max)
	if isinstance(args.synchtol, str):
		# Supply a particular tolerance to qualify as Synch
		synchtol = float(args.synchtol)
	else:
		synchtol = args.synchtol

	if not args.saving:
		# Which files to save? 
		#       Initial Conditions 'IC' 
		#       Synchronization trajectories 'Synch' 
		saving = []
	else:
		saving = saving.split(',')
	if args.plotting and len(args.plotting.split(','))>1:	
		plotting = args.plotting.split(',')
	else:
		plotting= None
	return Func, Func_net, DFunc, params, H, args.C, C_max, int(args.T_plot),args.matrix, A, G_name, nn, X0s, synchtol, cores, args.plotting, saving


def Simulate_Synch(inputs):
	# Simulates synchronization of chaotic oscillators
	#		on a network with adjacency matrix A in parallel
	# INPUTS:
	# 	A    - The chaotic map for the basic system from MSF 
	# 	X0   - The T=dt tangent map for the network from MSF
	#	eLH1 - Coupling matrix  e.g. delta_ij 
	#   T  	 - Maximum time frame for simulation of synchronization
	# 	Func_net - the network version of the chaotic oscillator map
	#   params- the parameters for the basic chaotic system
	#
	# OUTPUT:
	#	timetosynch - the time is takes for the network system to
	#		to synchronize to within synchtol 1E-5

	A, X0, eLH1, T, Func_net, params, synchtol= inputs
	a,b,c = params
# Simulate the dynamics
	with warnings.catch_warnings():
		#supress warnings from odeint due to step size issues
		warnings.simplefilter("ignore")
		X1 = odeint(Func_net, X0, range(T), args=(a,b,c,eLH1), tfirst=True, atol=1E-6, rtol=1E-6) #odeopts1)
	# fig, axs = plt.subplots(3,)
	# axs[0].plot(X1[:,0::3])
	# axs[1].plot(X1[:,1::3])
	# axs[2].plot(X1[:,2::3])
	# plt.show()
# Calculate how long it takes to synchronize to relative to the first node 
#			within an L1 norm of synchtol (should probably change to pdist)
	Distances = np.array([pdist(np.transpose(np.vstack([X1[i,::3],X1[i,1::3],X1[i,2::3]])),'minkowski', p=2.) for i in range(T)])
	errors = [max(Distances[i,:]) for i in range(T)]	
	timetosynch = next(i for i,x in enumerate(errors) if x<synchtol)
# Output the time it took to synchronize to within a tolerance
	return X1, timetosynch


def Simulate_ICs(inputs):
### Create a set of initial conditions for the Rossler Network System that 
#  		synchronize to within synchtol in under T steps

# use random to avoid issues with parallel threads using same randomization seed
	G_name, N, nn, eLH1, T_trans, T, Func, Func_net, params, synchtol, seed = inputs
	local_random = random.Random(seed)
# Simulate Oscillator to obtain a point near attractor
	if isinstance(eLH1, float):
		d = 1
	else:
		d = int(np.shape(eLH1)[0]/N)
	a,b,c = params
	y0 = list([local_random.random() for i in range(d)])
	##############################
	# May need to alter this for other systems without 3 parameters
	X = odeint(Func, y0, [0, T_trans], tfirst=True, args=(a,b,c), atol=1E-6, rtol=1E-6)
# Now set initial point near the attractor for the whole network of n oscillators
	S1 = np.zeros((d*N,))
	for i in range(d):
		S1[i::d] = X[-1,i]
# Make it so the oscillators do not start out synchronized...
	cnt=0
	X0s = np.zeros((nn,d*N))
	synchtimes= []
	skipped = 0
	while cnt<nn:
		# ADD PERTURBATIONS TO INITIAL CONDITIONS
		X0 = S1 + np.array([local_random.normalvariate(0,0.5) for i in range(d*N)])
		
	# Simulate the dynamics on the network with interaction
		with warnings.catch_warnings():
			#supress warnings from odeint due to step size issues
			warnings.simplefilter("ignore")
			X1 = odeint(Func_net, X0 ,range(T), args=(a,b,c, eLH1), tfirst=True, atol=1E-6, rtol=1E-6) #odeopts1)
	# Calculate how long it takes to synchronize to relative to the first node 
	#			within an L1 norm of synchtol (should probably change to pdist)
		Distances = np.array([pdist(np.transpose(np.vstack([X1[i,j::d] for j in range(d)])),'minkowski', p=2.) for i in range(T)])
		errors = [max(Distances[i,:]) for i in range(T)]
		if min(errors)<synchtol:
			timetosynch = next(i for i,x in enumerate(errors) if x<synchtol)
			synchtimes.append(timetosynch)
			X0s[cnt,:] = X0
			cnt += 1
		else:
			skipped += 1
		if skipped>5 and float(skipped/(skipped+cnt))>0.9:
			print('Not enough converging')
			break
#			print('skipped %s/%s potential IC due to divergence' % (skipped, skipped+cnt))
	# fig, axs = plt.subplots(d)
	# for i in range(N):
	# 	for j in range(d):
	# 		axs[j].plot(X1[:,d*i+j])
	# plt.show()
	return X0s, synchtimes, skipped/(skipped+cnt)


def Find_C(Interval, E, Cmax=False, printer=False):
	if Interval:
		Cmin = Interval[0] / E[1]
	else:
		return None
	#	print('Cmin=%s' % Cmin)

	if len(Interval)==1 and Cmax:
		if Cmin>Cmax:
			if printer:
				print('Not Synchronizable due to coupling restriction')
			return None
		else:
			return (Cmin+Cmax)/2.
#		print('Cmin=%s, Cmax=%s' % (Cmin,Cmax))
	elif len(Interval)>1:
		if Cmax:
			Cmax = min(Cmax, Interval[1] / E[-1])
		else:
			Cmax = Interval[1] / E[-1]
		if Cmin>Cmax:
			if printer:
				print('Not Synchronizable due to MSF analysis')
			return None
		else:
			return (Cmin+Cmax)/2.

if __name__=='__main__':

	args = get_input()
	Func, Func_net, DFunc, params, H, C, C_max, T_plot, matrix, A, G_name, nn, X0s, synchtol, cores, plotting, saving = process_input(args)

	N = np.shape(A)[0]			# number of nodes
	rowsums = np.sum(A,axis=0)
	edges = int(sum(rowsums))/2	# number of links
# Laplacian
	if not matrix:
		D = np.diag(rowsums)
		L = D-A
	elif matrix=='RW':
		L = np.identity(N) - np.dot(np.linalg.inv(D),A)
	E,V = np.linalg.eig(L)
	E = np.sort(np.real(E))
# Use MSF Analysis to optimize C if not provided
	if not C:
		if 'MSF' in plotting:
			Interval = PlotMSF(Func, DFunc, H, params, cores)
		else:
			# Use plotting='Attractor' to plot attractor
			Interval = MSF(Func, DFunc, H, params, plotting)
		print('Approx MSF interval is ', Interval)
		C = Find_C(Interval,E,C_max,True)
	else:
		C = float(C)
	print('C=%s' % C)
# Interaction term for identical synchronization model
# note that in the attractor network functions we use 
#     negative of this so G=-L as in Pecora/Carroll
	if isinstance(H, float):
		CLH = C*L 				  # For the simple case
	else:
		CLH = np.kron(C*L,H)      #Kronecker Tensor Product
	
#Simulate until a single (3D) point is relaxed onto the attractor 
	T_trans, T = [1000, 500] 	# Sim time for transit to attractor and synch

# If no Initial Conditions specified create a set of nn that will synchronize
#  		Save these initial conditions in a txt file input by user for later reuse
	if X0s is None:
		if nn>10:
			# If many runs, run in parallel chunks of 10 
			M = int(nn/10.)
			state = np.random.randint(2E8, size=M)
			pool = Pool(cores)
			results = pool.map(Simulate_ICs, [(G_name, N, min([nn,10]), CLH, T_trans, T, Func, Func_net, params, synchtol, state[i]) for i in range(M)])	
			pool.close()
		else:
			# If not many, run in parallel cases
			state = np.random.randint(2E8,size=nn)
			pool = Pool(cores)
			results = pool.map(Simulate_ICs, [(G_name, N, 1, CLH, T_trans, T, Func, Func_net, params, synchtol, state[i]) for i in range(nn)])	
			pool.close()
		# collect results and reshape
		X0s = np.array([i[0] for i in results]).reshape((nn,3*N))
		times = np.array([i[1] for i in results]).reshape((nn,))
		for i in results:
			# Track how many of the perturbed IC did not converge for debugging
			print('skipped %s potential IC due to divergence in thread' % i[2])
# non-parallelized
#		X0s = Simulate_ICs(G_name, N, nn, eLH1, T_trans, T, Func, Func_net, params, state)
# Use a pandas Data frame to save to csv for genfromtxt input later
		if 'ICs' in saving:	
			df = pd.DataFrame(X0s)
			IC_file = './InitialConditions/IC_in_Basin_%s_%s_%s.txt' % (G_name, N, nn)
			df.to_csv(IC_file, index=False)	
#	else:

	#If Synchronizing Initial Conditions are provided 
		# Run all instances in parallel on CPUs using Pool 
		# Use known X0s that will synchronize
	pool = Pool(cores)
	results = pool.map(Simulate_Synch, [(A, X0s[i,:], CLH, T, Func_net, params, synchtol) for i in range(nn)])	
	pool.close()
	times = np.array([i[1] for i in results if i])
	X = np.array([i[0][:,0:3*N:3] for i in results if i]).reshape((T,N))
	Y = np.array([i[0][:,1:3*N+1:3] for i in results if i]).reshape((T,N))
	Z = np.array([i[0][:,2:3*N+1:3] for i in results if i]).reshape((T,N))
	if 'Synch' in saving:
		df = pd.DataFrame(np.vstack([X,Y,Z]))
		Synch_file = './Synchronization/Synch_%s_%s_%s.txt' % (G_name, N, nn)
		df.to_csv(Synch_file, index=False)	
#		print('Percent synch = %s' % (np.round(len(times)/len(results)*100.,2)), 'Ave(timesynch) = %s' % np.round(np.mean(times),2))
	if 'hist' in plotting:
		# plot histogram of times to synch for X0s
		plt.hist(times)	
		plt.xlim(0,max(times)+20)
#		plt.ylim(0,int(nn/3))
		plt.annotate('%s_%s_%s synch = %s Ave(time) = %s' % (G_name, N, edges, np.round(len(times)/len(results)*100.,2), np.round(np.mean(times),2)), xy=(2, 1), xytext=(20, int(nn/4)+5))
		plt.draw()
		plt.savefig("".join(['./figures/', G_name,'_', str(N), '_',str(edges),'hist.png']))

	# #Notice how all oscillators converge in this situation (we know they should
	# #thanks to the MSF)
	if 'Synch' in plotting:
#		plt.style.use('dark_background')
		fig, axs = plt.subplots(3,)
		for i in range(N):
			axs[0].plot(X[:T_plot,i])
			axs[1].plot(Y[:T_plot,i])
			axs[2].plot(Z[:T_plot,i])
		axs[0].set_ylabel('x')
		axs[1].set_ylabel('y')
		axs[2].set_ylabel('z')
		axs[0].set_xlabel('t')
		axs[1].set_xlabel('t')
		axs[2].set_xlabel('t')
#		axs[0].legend(['0','1','2','3','4','5'])
		plt.show()