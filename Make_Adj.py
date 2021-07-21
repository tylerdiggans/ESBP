import random, argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from itertools import combinations

def get_input():
	parser = argparse.ArgumentParser()
	
	#------------------Model Parameters----------------------------
	parser.add_argument('--N', default=None, 
						help='Number of nodes (oscillators)')
	parser.add_argument('--nxType', default='complete_graph', 
						help='See documentation for networkx generators      ...\
						https://networkx.org/documentation/networkx-1.10/    ...\
						reference/generators.html OR \
						Choose Adj for an input adjacency matrix')
	parser.add_argument('--parameters', default=None, 
						help='See documentation for parameters for networkx   ...\
						generators https://networkx.org/documentation/		  ...\
						networkx-1.10/reference/generators.html')
	
	parser.add_argument('--added_lines', default=None, 
						help='"add" or "remove" or \
						change to have added lines of code commented')

	#------------------Plotting Parameters----------------------------
	parser.add_argument('--plotting', default=False, 
						help='Used to suppress plotting functionality')
	parser.add_argument('--pathname', default='./', 
						help='filename for saving adjacency matrix as txt')
	parser.add_argument('--filename', default='Adjacency.txt', 
						help='filename for saving adjacency matrix as txt')

	args = parser.parse_args()
	return args         

def process_input(args):
	# Used for adding or removing edges from an 
	#    Adjacency matrix that was saved using to_csv
	if args.nxType=='Adjacency':
		A=pd.DataFrame(np.genfromtxt(args.parameters, skip_header=1, delimiter=','))
		G = nx.convert_matrix.from_pandas_adjacency(A)
	# Deal with special nxTypes based on the documentation as needed
	elif args.nxType=='random_regular_graph':
		# N is the number of nodes and parameters are found in docs
		G = eval("".join(['nx.', args.nxType, '(', args.parameters,', n=', str(args.N), ')']))		
	elif args.nxType=='newman_watts_strogatz_graph':		
		# N nodes, k-cycle with random connection prop p
		k, p = args.parameters.split(',')
		G = eval("".join(['nx.', args.nxType, '(', str(args.N), ', ', str(k), ', ', str(p),')']))		
	elif not args.parameters:
		# Most other nxTypes have similar structures
		if not args.N:
			# Some do not require any inputs
			G = eval("".join(['nx.', args.nxType, '()']))	
		else:
			# Some only require the order N
			G = eval("".join(['nx.', args.nxType, '(', str(args.N),')']))	
	else:
		# Others just use two parameters (not necessarily N),
		#    those with a single parameter can be used here by 
		#    including N as the first parameter
		r, h = args.parameters.split(',')
		G = eval("".join(['nx.', args.nxType, '(', str(r),', ', str(h), ')']))	
	if args.N:
		N = int(args.N)
	else:
		N = None

	# Sometimes it is helpful to make an adjacency matrix of a perturbation
	#     to a known graph, here we can use the input added_lines
	if args.added_lines:
		# This was developed in an ad-hoc manner and can be altered 
		#     however is useful to the user
		added_lines = args.added_lines.split(' ')
		if added_lines[0] == 'add':
			# Add a random edge
			possibles = list(set(combinations(range(G.order()),2))-set(G.edges()))
			index = np.random.choice(len(possibles), int(added_lines[1]),replace=False)
			for ind in index:
				G.add_edge(possibles[int(ind)][0], possibles[int(ind)][1])
		elif added_lines[0] == 'remove':
			# remove a random edge
			index = np.random.choice(len(G.edges()), int(added_lines[1]),replace=False)
			for ind in index:
				e = [u for u in G.edges()][int(ind)]
				G.remove_edge(e[0], e[1])
		elif added_lines[0] == 'pendant':
			# Add a new vertex as a pendant vertex of a random vertex
			N = G.order()
			# Add M pendants
			M = int(added_lines[1])
			if M<=N:
				parents = np.random.choice(range(N), M, replace=False)
			else:
				parents = np.random.choice(range(N), M)				
			for i in range(M):
				G.add_node(N+i)
				G.add_edge(parents[i],N+i)
			# update the order of the graph
			N += M			
		elif added_lines[0]=='pasteK3':
			# example of a more elaborate case where we past copies of K3
			N = G.order()
			deg = np.array([d for n,d in G.degree()])
			print(deg)
			ind = np.where(deg==1)[0]
			P = len(ind)
			print(ind, P)
			M = int(added_lines[1])
			parents = np.random.choice(ind, M, replace=False)
			ancestors = np.array([[u for u in G.neighbors(parents[i])] for i in range(M)]).flatten()
			for i in range(M):
				G.add_nodes_from(range(N+i*2,N+(i+1)*2))
				G.add_edges_from([(ancestors[i], j) for j in range(N+i*2,N+(i+1)*2)]+
									[(parents[i], N+i*2), (N+i*2,N+i*2+1), (N+i*2+1,parents[i])])
			N += 3*M			
		elif added_lines[0] == 'multiply':
			# Use multiples of a graph and past together (contract) nodes
			H = deepcopy(G)
			for m in range(1,int(added_lines[1])):
				# create a mapping of old nodes to new labels
				count = m*G.order()
				d = {} #Empty dictionary to add values into
				for i in G.nodes():
					d[i] = count
					count+=1
				K = nx.relabel_nodes(G, d)			
				H.add_nodes_from(K.nodes())
				H.add_edges_from(K.edges())
			# now we need to contract some nodes
			for m in range(1,int(added_lines[1])):
				H = nx.contracted_nodes(H, 0, m*G.order())
#			G = deepcopy(H)
			G = nx.contracted_nodes(H,1,G.order())
		elif added_lines[0] == 'cycle':
			# Make a central cycle of contracted copies of a 
			#    single graph
			H = deepcopy(G)
			for m in range(1,int(added_lines[1])):
				# create a mapping of old nodes to new labels
				count = m*G.order()
				d = {} #Empty dictionary to add values into
				for i in G.nodes():
					d[i] = count
					count+=1
				K = nx.relabel_nodes(G, d)			
				H.add_nodes_from(K.nodes())
				H.add_edges_from(K.edges())
			# now we need to create a central cycle
			for m in range(int(added_lines[1])-1):
				H.add_edge(m*G.order(), (m+1)*G.order(),)
			H.add_edge((m+1)*G.order(),0)
			G = deepcopy(H)
	savefile = "".join([str(args.pathname), str(args.filename)])

	return N, G, args.plotting, savefile


if __name__=='__main__':
	args = get_input()
	N, G, plotting, savefile = process_input(args)
	df = nx.convert_matrix.to_pandas_adjacency(G)
	df.to_csv(savefile, index=False)
	if plotting:
		nx.draw(G)	
		plt.show()