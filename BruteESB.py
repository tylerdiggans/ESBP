import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import comb
from itertools import *
from copy import deepcopy
from multiprocessing import Pool, cpu_count
# Import some functions from my own py file
from MSFTools import MSF
from Utilities import Find_C, SeparateEm, Plot3D, MakePlot3DBackbone, find_all_cycles

def get_input():
	parser = argparse.ArgumentParser()
	#------------------Default Parameters--------------------------
	parser.add_argument('--system', default=None, 
						help='Basic chaotic attractor defaults to Rossler')
	parser.add_argument('--params', default='0.2, 0.2, 9.0', 
						help='parameters for basic chaotic attractor')
	parser.add_argument('--H', default='x', 
						help='coupling matrix defaults to x variable (use x,y,z, or 3x3 matrix)')
	parser.add_argument('--Interval', default='0.186,4.614', 
						help='The interval from MSF for normalized coupling eigs')
	parser.add_argument('--C_max', default=None, 
						help='coupling strength limitation')
	parser.add_argument('--matrix', default=None, 
						help='change the matrix to RW')
	parser.add_argument('--Adj', default=None, 
						help='txt filename containing an adjacency matrix \
								saved using pandas.to_csv()')
#------------------Plotting Parameters----------------------------
	parser.add_argument('--cores', default=None, 
						help='Howmany cores to use for pool')
	parser.add_argument('--Positions', default=None, 
						help='plotting positions')
	parser.add_argument('--plotting', default=True, 
						help='Used to enable/suppress plotting functionality')
	parser.add_argument('--pathname', default='./Backbones/', 
						help='filename for saving adjacency matrix as txt')
	parser.add_argument('--filename', default=None, 
						help='filename for saving adjacency matrix as txt')
	args = parser.parse_args()
	return args         

def process_input(args):
	##########################################################
	##########################################################
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
		from Attractor_Functions import rossler, DF_rossler
		Func = rossler
		DFunc = DF_rossler
	elif args.system == 'Lorenz':
		from Attractor_Functions import lorenz, DF_lorenz
		Func = lorenz
		DFunc = DF_lorenz
	elif args.system == 'Chens':
		from Attractor_Functions import chens, DF_chens
		Func = chens
		DFunc = DF_chens
	elif args.system == 'Harmonic':
		from Attractor_Functions import harm, DF_harm
		Func = harm
		DFunc = DF_harm
		H = H[:2,:2]
# Process the parameters for the chaotic oscillators
	params = tuple([float(eval(x)) for x in args.params.split(',')]) 

# If interval from MSF is supplied skip that portion, 
#			otherwise approximate using MSF
	if args.Interval:
		Interval = [float(i) for i in args.Interval.split(',')]
	else:
		Interval = None

# Optional setting maximum coupling strength to force interval (Harmonic)
	if not args.C_max:
		C_max = None
	else:
		C_max = float(args.C_max)

# Extract the name of the graph and create a save file for the backbone
	if not args.filename:
		G_name = args.Adj.split('/')[-1].split('.')[0]			
		filename = ''.join([str(G_name),'.txt'])
	else:
		filename = str(args.filename)
		G_name = filename.split('.')[0]	
	savefile = "".join([str(args.pathname), filename])	

# Optional assignment of a number of cores to the current trial
	if args.cores:
		cores = int(args.cores)
	else:
		cores = cpu_count()
	if not args.Positions:
		X = None
	else:
		X = np.loadtxt(args.Positions)
	return Func, DFunc, params, H, Interval, C_max, args.matrix, args.Adj, G_name, cores, X, args.plotting, savefile
 

def powerset(iterable,N_min,N_max):
 	# This is used to obtain subsets of the powerset of the edge set
#    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
#    "powerset([1,2,3], 1,2) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(N_min,N_max+1))


def Find_ratio(inputs):
	# Used to parallelize the use of Find_C

	# Use the Interval from the MSF and an edgeset ES to create a 
	#  network and see if its eigenvalues satisfy MSF_Ratio
	N, ES, Interval, C_max, matrix = inputs
	K = nx.Graph()
	K.add_nodes_from(range(N))
	K.add_edges_from(ES)
	# Must be connected to synchronize
	if nx.is_connected(K):
		if not matrix:
			E = nx.laplacian_spectrum(K)
		elif matrix=='RW':	
			E = nx.normalized_laplacian_spectrum(K)
		# Find the optimum coupling strength for synch
		C = Find_C(Interval, E, C_max)
		# if C then the eigenvalues fit in Kspan, so just optimize ratio
		if C:
			new_ratio = E[-1] / E[1]
		else:
			new_ratio=None
	else:
		new_ratio=None
	return K, new_ratio


if __name__=='__main__':
	args = get_input()
	Func, DFunc, params, H, Interval, C_max, matrix, Adj, G_name, cores, X, plotting, savefile = process_input(args)
# An adjacency matrix in a txt file saved using pandas to_csv()
	A = np.genfromtxt(Adj, skip_header=1, delimiter=',')
	G = nx.from_numpy_matrix(A)					# Create networkx.Graph
	N = G.order()								# Number of vertices

# Dark plotting theme
	dark = False

	# Verify synchronization for original graph G
	if nx.is_connected(G):
		if not matrix:
			E = nx.laplacian_spectrum(G)
		elif matrix=='RW':
			E = nx.normalized_laplacian_spectrum(G)
	else:
		print('Original Graph not connected start over')

	if not Interval:
		# Use MSF Analysis to choose a safe C if not provided
		# plotting options are 'MSF' and 'Attractor'
		Interval = MSF(Func, DFunc, H, params, 'plotting')	
	C = Find_C(Interval, E, C_max, False)		
	if not C:
		raise Exception('Original graph not Synchronizable')
	# If C exists, then the original graph is synchronizable

	# This creates the powerset of subsets of the edge list 
	#		of length between N-1 (tree) and M-1 one less
	# Start from N-1 (spanning trees) and go until we find self-backbone
	cnt, M = [N-1, nx.number_of_edges(G)]
	notdone = True
	while notdone and cnt<=M:
		print('Calculating for %s graphs with %s edges' % (comb(M,cnt),cnt))
		edgesets = powerset(G.edges(), cnt,cnt)
		T = int(np.floor(comb(M,cnt)/1000))
		# 	Compute the MSF ratios of various edgesets in T groups 
		#		of 1000 to prevent overloading pool and being 
		#       able to exit the routine before checking ALL 
		#       of powerset(G.edges, cnt,cnt)
		for t in range(T):
			if not notdone:
				break
			# Take the next 1000 edgesets
			Some_edgesets = list(next(edgesets) for _ in range(1000))
			pool = Pool(cores)
			results = pool.map(Find_ratio, [(N, ES, Interval, C_max, matrix) for ES in Some_edgesets])	
			pool.close()
			# Save any that satisfy MSF
			Output = np.array([(K,r) for K,r in results if r], dtype=object)
			if len(Output) > 0:
				if Output[:,1].any():
					# Save the smallest ratio of those that have been found
					H, new_ratio = Output[np.argmin(Output[:,1]),:]
					notdone= False
		# Finish off the remainder for cnt edges if none found yet
		if notdone:
			pool = Pool(cores)
			results = pool.map(Find_ratio, [(N, ES, Interval, C_max, matrix) for ES in list(next(edgesets) for _ in range(comb(M,cnt) % 1000))])	
			pool.close()
			Output = np.array([(K,r) for K,r in results if r], dtype=object)
			if len(Output) > 0:
				if Output[:,1].any():
					H, new_ratio = Output[np.argmin(Output[:,1]),:]
					notdone= False
		# increment the number of edges being selected
		cnt += 1
	if not matrix:
		E = nx.laplacian_spectrum(H)
	elif matrix=='RW':
		E = nx.normalized_laplacian_spectrum(H)
	new_ratio = E[-1]/E[1]
	print('Edges in G: %s \nEdges in H: %s\n' % (nx.number_of_edges(G), nx.number_of_edges(H)))
	if savefile:
		df = nx.convert_matrix.to_pandas_adjacency(H)
		df.to_csv(savefile, index=False)

	# Find low energy positions for plotting (Generation, Nu) 
	# X is used as the variable for Positions
	if X is None:
		X = SeparateEm(N, nx.adjacency_matrix(H), 40, 400, 1., N)
		np.savetxt('./Backbones/Current_Positions.txt', X)		
# plot the interactive backbone for manipulation
	fig, ax = Plot3D(H,X,  width=0.75, dark=dark)
	ax.text(0, 0, 0, '$\lambda_N/\lambda_2=$%s' % np.round(new_ratio,2))
	plt.show()

# Make new GIF with correct ending positions
	if plotting=='GIF':
		ordered_edges = list(set(G.edges()) - set(H.edges()))
		MakePlot3DBackbone(G_name, N, G, X, ordered_edges)

# Find all the cycles of the backbone
	if plotting == 'Cycles':
		# cycles = find_all_cycles(H, source=None, cycle_length_limit=N)
		# lengths = [len(i) for i in cycles]
		if dark:
			plt.style.use('dark_background')
		bas_lens = [len(i) for i in nx.cycle_basis(H)]
		fig, ax_cycle = plt.subplots(1, figsize=(10, 10))
		ax_cycle.hist(bas_lens, bins = range(N))
		ax_cycle.set_xlabel('Basis Cycle Lengths')
		ax_cycle.set_ylabel('Count')
		ax_cycle.set_title('Backbone Base Cycles')
		plt.savefig("".join(['./figures/BackboneCycles_', str(G_name),'.png']))
		plt.close()