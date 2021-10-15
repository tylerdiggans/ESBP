import random, argparse, os, sys, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as time
from copy import deepcopy
from PIL import Image, ImageDraw
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint
from scipy.spatial.distance import pdist
from scipy.linalg import eigh as largest_eigh
#from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
# Import some functions from my own py file
from MSFTools import MSF
from Utilities import Find_C, SeparateEm, Plot3D, MakePlot3DBackbone, find_all_cycles


def get_input():
	parser = argparse.ArgumentParser()
	#------------------Default Parameters--------------------------
	parser.add_argument('--H', default='x', 
						help='coupling matrix defaults to x variable (use x,y,z, or 3x3 matrix)')
	parser.add_argument('--C_max', default=None, 
						help='coupling strength limitation')
	parser.add_argument('--system', default=None, 
						help='Basic chaotic attractor defaults to Rossler')
	parser.add_argument('--matrix', default=None, 
						help='change the matrix to RW')
	parser.add_argument('--params', default='0.2, 0.2, 9.0', 
						help='parameters for basic chaotic attractor')	
	parser.add_argument('--cores', default=None, 
						help='Howmany cores to use for pool')	
#------------------Model Parameters----------------------------
	parser.add_argument('--Adj', default=None, 
						help='txt filename containing an adjacency matrix \
								saved using pandas.to_csv()')
	parser.add_argument('--Interval', default='0.186,4.614', 
						help='The interval from MSF for normalized coupling eigs')
#------------------Plotting Parameters----------------------------
	parser.add_argument('--Positions', default=None, 
						help='plotting positions')
	parser.add_argument('--plotting', default=True, 
						help='Used to enable/suppress plotting functionality')
	parser.add_argument('--pathname', default='./', 
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
		from Attractor_Functions import lorenz, DF_lorenz
		Func = lorenz
		DFunc = DF_lorenz
	elif args.system == 'Chens':
		from Attractor_Functions import chens, chens_net_ll, DF_chens
		Func = chens
		Func_net = chens
		DFunc = DF_chens
	if args.cores:
		cores = int(args.cores)
	else:
		cores = cpu_count()
# If interval from MSF is supplied skip that portion
# THIS SAVES A LOT OF TIME, AND SHOULD ALMOST ALWAYS BE USED
	if args.Interval:
		Interval = [float(i) for i in args.Interval.split(',')]
	else:
		Interval = NoneS
	if not args.C_max:
		# Impose a global coupling strength Maximum?
		C_max = None
	else:
		C_max = float(args.C_max)
	if not args.filename:
		# Rip the name of the graph being used for auto saving naming
		G_name = args.Adj.split('/')[-1].split('.')[0]
		filename = ''.join([str(G_name),'.txt'])
	else:
		filename = str(args.filename)
	if not args.Positions:
		# Supply node positions if you want to use the same embedding 
		#     as previous runs of similar cases
		X = None
	else:
		X = np.loadtxt(args.Positions)
	savefile = "".join([str(args.pathname), filename])	
	return args.Adj, args.matrix, Func, DFunc, H, C_max, params, Interval, cores, X, args.plotting, savefile
 


if __name__=='__main__':
	start = time.time()
	args = get_input()
	Adj, matrix, Func, DFunc, H, C_max, params, Interval, cores, X, plotting, savefile = process_input(args)
# An adjacency matrix in a txt file saved using pandas to_csv()
	A = np.genfromtxt(Adj, skip_header=1, delimiter=',')
	G = nx.from_numpy_matrix(A)					# Create networkx.Graph
	G_name = Adj.split('/')[-1].split('.')[0]	# Extract the filename for saving
	N = G.order()								# Number of vertices
#	dark = True
	dark = False  # plot style

# Verify synchronization for original graph G
	if nx.is_connected(G):
		if not matrix:
			E = nx.laplacian_spectrum(G)
		elif matrix=='RW':
			E = nx.normalized_laplacian_spectrum(G)
	else:
		print('Original Graph not connected start over')
		sys.exit()
	if not Interval:
		# Use MSF Analysis to obtain the K_span Interval
		# plotting options are 'MSF' and 'Attractor'
		Interval = MSF(Func, DFunc, H, params, 'plotting')	
	# Find a coupling strength that synchronizes if exists
	C = Find_C(Interval, E, C_max, False)		
	print('Using C=%s' % C)
	if not C:
		raise Exception('Original graph not Synchronizable')

	# If C exists, then the original graph is synchronizable
	ratio = E[-1]/E[1]
	H = deepcopy(G)
#	print('G has %s edges' % len(G.edges()))
	ordered_edges = []
	M = nx.number_of_edges(G)
	# At best we find a spanning tree
	for m in range(M-G.order()+1):
		the_edge=None
		if not matrix:                      
			L = nx.laplacian_matrix(H).todense()
		elif matrix=='RW':
			L = nx.normalized_laplacian_matrix(H).todense()
		# From Hagberg08 use V_N to choose greedy edge (not always optimal)
		lambda_N, V_N = largest_eigh(L, eigvals=(N-1,N-1))        
		diffs = np.array([[i, j, abs(V_N[i]-V_N[j])] for (i,j) in H.edges()],dtype=object)
		diffs=diffs[diffs[:,2].argsort()]
		cnt = np.shape(diffs)[0]-1
		while not the_edge and cnt>=0:
			# try E-e until one satisfies MSF
			K = deepcopy(H)
			e = (diffs[cnt,0], diffs[cnt,1])
			cnt -= 1
			K.remove_edge(e[0],e[1])
			if nx.is_connected(K):
				if not matrix:
					E = nx.laplacian_spectrum(K)
				elif matrix=='RW':
					E = nx.normalized_laplacian_spectrum(K)
				C = Find_C(Interval, E, C_max)
				if C:
					the_edge = deepcopy(e)
		if the_edge:
			ordered_edges.append(the_edge)
			H.remove_edge(the_edge[0], the_edge[1])
		else:
			# if none satisfy R<RMSF, then exit
			break
	if not matrix:
		E = nx.laplacian_spectrum(H)
	elif matrix=='RW':
		E = nx.normalized_laplacian_spectrum(H)
	new_ratio = E[-1]/E[1]
	end = time.time()
	print(end-start)	
	print(nx.number_of_edges(G), nx.number_of_edges(H))
	if savefile:
		# Print Backbone Adjacency into ./Backbones folder
		df = nx.convert_matrix.to_pandas_adjacency(H)
		df.to_csv(savefile, index=False)

#	print('FinalRatio=%s' % new_ratio)
	# Find low energy positions for plotting (Generation, Nu) 
	if X is None:
		X = SeparateEm(N, nx.adjacency_matrix(H), 40, 400, 1., N)
		np.savetxt('./Backbones/Current_Positions.txt', X)		
# Make new GIF with correct ending positions

#	MakePlot3DBackbone(G_name, N, G, X, ordered_edges, dark)
# Find all the cycles of the backbone
	if plotting=='Cycles':
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
		ax_cycle.set_title('Greedy Cycle Basis', size=22)
		plt.savefig("".join(['./figures/BackboneCycles_', str(G_name),'.png']), bbox_inches='tight')
		plt.close()

# plot the interactive backbone for manipulation
	fig, ax = Plot3D(H, X, width=1.0, dark=dark)
	ax.text(0, 0, 0, '$\lambda_N/\lambda_2=$%s' % np.round(new_ratio,2))
	plt.show()
	# E = nx.laplacian_spectrum(H)
	# C = Find_C(Interval, E)
	# if not C:
	# 	print('Not Synchronizable due to MSF analysis')
	# else:
	# 	df = nx.convert_matrix.to_pandas_adjacency(H)
	# 	df.to_csv(savefile, index=False)
	#os.system('python Synch_net_wMSF.py --Adj='+savefile+' --parallel=100 --C='+str(C))