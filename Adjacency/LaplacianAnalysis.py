import random, argparse, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint
from scipy.spatial.distance import pdist
from Net3DSee import SeparateEm, Plot3D, Plot2D, MakePlot3DBackbone

def get_input():
	parser = argparse.ArgumentParser()
	
	#------------------Model Parameters----------------------------
	parser.add_argument('--Adj', default=None, 
						help='txt filename containing an adjacency matrix \
								saved using pandas.to_csv()')
#------------------Plotting Parameters----------------------------
	parser.add_argument('--plotting', default=True, 
						help='Used to suppress plotting functionality')

	parser.add_argument('--positions', default=None, 
						help='Used for plotting functionality')

	args = parser.parse_args()
	return args         

def process_input(args):
# An adjacency matrix must be supplied
	A = np.genfromtxt(args.Adj, skip_header=1, delimiter=',')
	N = np.shape(A)[0]
#	savefile = "".join([str(args.pathname), str(args.filename)])
	return N, A, args.positions, args.plotting
 


#############################################################
###https://gist.github.com/joe-jordan/6548029#file-cycles-py#
#############################################################
def find_all_cycles(G, source=None, cycle_length_limit=None):
	"""forked from networkx dfs_edges function. Assumes nodes are integers, or at least
	types which work with min() and > ."""
	if source is None:
		# produce edges for all components
		nodes=[list(i)[0] for i in nx.connected_components(G)]
	else:
		# produce edges for components with source
		nodes=[source]
	# extra variables for cycle detection:
	cycle_stack = []
	output_cycles = set()
	
	def get_hashable_cycle(cycle):
		"""cycle as a tuple in a deterministic order."""
		m = min(cycle)
		mi = cycle.index(m)
		mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
		if cycle[mi-1] > cycle[mi_plus_1]:
			result = cycle[mi:] + cycle[:mi]
		else:
			result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
		return tuple(result)
	
	for start in nodes:
		if start in cycle_stack:
			continue
		cycle_stack.append(start)
		
		stack = [(start,iter(G[start]))]
		while stack:
			parent,children = stack[-1]
			try:
				child = next(children)
				
				if child not in cycle_stack:
					cycle_stack.append(child)
					stack.append((child,iter(G[child])))
				else:
					i = cycle_stack.index(child)
					if i < len(cycle_stack) - 2: 
					  output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
				
			except StopIteration:
				stack.pop()
				cycle_stack.pop()
	
	return [list(i) for i in output_cycles]

if __name__=='__main__':

	args = get_input()
	N, A, positions, plotting = process_input(args)

	# A = np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
	# N = 4
# Laplacian
	G = nx.Graph(A)
	if positions is None:
		pos = nx.spring_layout(G)
		# pos = SeparateEm(N, A, 40, 500, 1., N)
		# np.savetxt('positions2.txt', pos)
	else:
		pos = np.loadtxt(positions)

		#fig, ax = Plot2D(G, pos, width=1, dark=False) 
	nx.draw(G)
	# ax.text(1, 0, 0, '$\lambda_N/\lambda_2=18.75$')
	plt.show()

	D = np.diag(np.sum(A,axis=0))
	L = D-A
	E,V = np.linalg.eig(L)
	order = np.argsort(E)
	E = E[order]
	V = np.round(V[:,order], 3)
	print(E)
	print('Eigenratio=%s' % (E[-1]/E[1]))
# # random walk laplacian
# 	Lrw = np.identity(N) - np.dot(np.linalg.inv(D),A)
# 	Erw,Vrw = np.linalg.eig(Lrw)
# 	order = np.argsort(Erw)
# 	Erw = Erw[order]
# 	Vrw = np.round(Vrw[:,order], 3)
# 	print(Erw)
# 	print(np.dot(np.dot(Vrw, np.diag(Erw)),np.linalg.inv(Vrw)))
# # Normalized Laplacian
# 	D_half = np.diag([1./np.sqrt(D[i,i]) for i in range(N)])
# 	nL = np.identity(N) - np.dot(np.dot(D_half,A),D_half)
# 	nE, nV = np.linalg.eig(nL)
# 	order = np.argsort(nE)
# 	nE = nE[order]
# 	nV = np.round(nV[:,order],3)
# #	print(E,'\n',nE,'\n',Erw, '\n', np.linalg.eig(A))

# #	print(V,'\n', nV,'\n',Vrw)

# 	if nV[0,0]<0:
# 		nV *= -1
# 	if V[0,0]<0:
# 		V *= -1	

# 	fig1, ax1s = plt.subplots(2,figsize=(N,8))

# 	ax1s[0].plot(range(N), E, 'd')
# 	ax1s[0].grid()

# #	ax.plot(range(N), 2.*np.sort(E)/(n-1),'*')
# 	ax1s[1].plot(range(N), nE, '+')
# 	ax1s[1].grid()

# 	fig2, ax2s = plt.subplots(2,figsize=(8,8))
# 	for i in range(N):
# 		ax2s[0].plot(range(N), V[:,i])
# 		ax2s[1].plot(range(N), nV[:,i])	
# 	ax2s[0].grid()
# 	ax2s[1].grid()
# 	ax2s[0].legend(range(N),loc='upper right')
# 	ax2s[1].legend(range(N),loc='upper right')
	
# 	plt.show()
