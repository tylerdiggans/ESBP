import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time as time
from copy import deepcopy
from multiprocessing import Pool, cpu_count
# Import some functions from my own py file
from MSFTools import MSF
from Utilities import Find_C, SeparateEm, Plot3D, Plot2D, MakePlot3DBackbone, find_all_cycles


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
						help='coupling strength limitation to force interval')
	parser.add_argument('--matrix', default=None, 
						help='change the matrix from L to L_RW')
	parser.add_argument('--Adj', default=None, 
						help='txt filename containing an adjacency matrix \
								saved using pandas.to_csv()')
#------------------Plotting Parameters----------------------------
	parser.add_argument('--cores', default=None, 
						help='Howmany cores to use for pool')	
	parser.add_argument('--Positions', default=None, 
						help='plotting positions')
	parser.add_argument('--plotting', default=None, 
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

# Process the parameters for the chaotic oscillators
	params = tuple([float(eval(x)) for x in args.params.split(',')]) 

# Import the chaotic attractor being used
	if not args.system:
		from Attractor_Functions import rossler, DF_rossler
		Func = rossler
		DFunc = DF_rossler
	elif args.system == 'Lorenz':
# If Lorenz, overwrite with Lorenz parameters to make easy
		params = (10.,28.,8./3.)
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

# If interval from MSF is supplied skip that portion
	if args.Interval:
		Interval = [float(i) for i in args.Interval.split(',')]
	else:
		Interval = None    		# Leads to using MSF to approximate

# Optional Provide a limit on coupling strength for Harmonic
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

# Specify a number of cores to dedicate to this run
	if args.cores:
		cores = int(args.cores)
	else:
		cores = cpu_count()
	if not args.Positions:
		X = None
	else:
		X = np.loadtxt(args.Positions)
	return Func, DFunc, params, H, Interval, C_max, args.matrix, args.Adj, G_name, cores, X, args.plotting, savefile
 

if __name__=='__main__':

# The Hyrbid Exhaustive Greedy Decision algorithm for obtaining a GESB
	start = time.time()
	args = get_input()
	Func, DFunc, params, H, Interval, C_max, matrix, Adj, G_name, cores, X, plotting, savefile = process_input(args)

# An adjacency matrix in a txt file saved using pandas to_csv()
	A = np.genfromtxt(Adj, skip_header=1, delimiter=',')
	G = nx.from_numpy_matrix(A)					# Create networkx.Graph
	N = G.order()								# Number of vertices

# Dark plotting theme
#	dark = True
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
		# Use MSF Analysis to obtain the K_span Interval
		# plotting option can be changed to 'MSF' or 'Attractor'
		Interval = MSF(Func, DFunc, H, params, 'plotting')	
	C = Find_C(Interval, E, C_max, False)		
	if not C:
		raise Exception('Original graph not Synchronizable')
	# If C exists, then the original graph is synchronizable

	# Make a copy to alter through algorithm
	H = deepcopy(G)
	ordered_edges = []
	# At best we find a spanning tree (N+1)
	for m in range(nx.number_of_edges(G)-N+1):
		old_ratio = np.inf
		the_edge = None				# To be chosen
		# Check MSF ratio for each edge removal and take lowest
		for e in H.edges():
			K = deepcopy(H)			# Make another copy to test edge removal
			K.remove_edge(e[0],e[1])
			if nx.is_connected(K):
				if not matrix:						
					E = nx.laplacian_spectrum(K)
				elif matrix=='RW':
					E = nx.normalized_laplacian_spectrum(K)
				C = Find_C(Interval, E, C_max)
				# If C exists, then the network satisfies MSF
				if C:
					new_ratio = E[-1] / E[1]
					# Make sure we use the best of what is available regardless of old span
					if new_ratio <= old_ratio:
						the_edge = deepcopy(e)
						old_ratio = deepcopy(new_ratio)
		if the_edge:
			ordered_edges.append(the_edge)
			H.remove_edge(the_edge[0], the_edge[1])
		else:
			break
	if not matrix:
		E = nx.laplacian_spectrum(H)
	elif matrix=='RW':
		E = nx.normalized_laplacian_spectrum(H)
	new_ratio = E[-1]/E[1]
	end = time.time()
	print(end-start)
	print('Edges in G: %s \nEdges in H: %s\n' % (nx.number_of_edges(G), nx.number_of_edges(H)))
	if savefile:
		# Print Backbone Adjacency into ./Backbones folder
		df = nx.convert_matrix.to_pandas_adjacency(H)
		df.to_csv(savefile, index=False)

# plot the interactive backbone for manipulation otherwise
	# Find low energy positions for plotting (Generation, Nu) 
	if X is None:
		X = SeparateEm(N, nx.adjacency_matrix(H), 40, 400, 1., N)
		np.savetxt('./Backbones/Current_Positions.txt', X)		
# plot the interactive backbone for manipulation
	if np.shape(H)[0]==3:
		fig, ax = Plot3D(H, X, width=1.0, dark=dark)
		ax.text(0, 0, 0, '$\lambda_N/\lambda_2=$%s' % np.round(new_ratio,2))
	else:
		fig, ax = Plot2D(H, X, width=1.0, dark=dark)
		ax.text(0.1, 0.05, '$\lambda_N/\lambda_2=$%s' % np.round(new_ratio,2))
	plt.show()
	if plotting == 'GIF':
	# Make new GIF with correct ending positions
		MakePlot3DBackbone(G_name, N, G, X, ordered_edges, dark)

# Find all the cycles of the backbone
	if plotting == 'Cycles':
		if dark:
			plt.style.use('dark_background')
		# cycles = find_all_cycles(H, source=None, cycle_length_limit=N)
		# lengths = [len(i) for i in cycles]
		bas_lens = [len(i) for i in nx.cycle_basis(H)]
		plt.rc('axes', labelsize=22)
		plt.rc('xtick', labelsize=16)
		plt.rc('ytick', labelsize=16)
		fig, ax_cycle = plt.subplots(1, figsize=(8, 4))
		# ax_cycle[0].hist(lengths, bins = range(N))
		# ax_cycle[0].set_xlabel('All Cycle Lengths')
		# ax_cycle[0].set_ylabel('Count')
		# ax_cycle[0].set_title('Backbone Cycles')
		ax_cycle.hist(bas_lens, bins = range(1,25))
		ax_cycle.set_xlabel('Cycle Length ($l$)')
		ax_cycle.set_ylabel('Count')
		ax_cycle.set_title('Hybrid Cycle Basis', size=22)
		plt.savefig("".join(['./figures/BackboneCycles_', str(G_name),'.png']), bbox_inches='tight')
		plt.close()
