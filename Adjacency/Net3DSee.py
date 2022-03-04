import random, argparse, warnings, struct, io
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint
from scipy.spatial.distance import pdist

def get_input():
	parser = argparse.ArgumentParser()
	
	#------------------Model Parameters----------------------------
	parser.add_argument('--Adj', default=None, 
						help='txt filename containing an adjacency matrix \
								saved using pandas.to_csv()')
	parser.add_argument('--Backbone', default=None, 
						help='txt filename containing an adjacency matrix \
								of a backbone saved using pandas.to_csv()')
	parser.add_argument('--Positions', default=None, 
						help='txt filename containing positions for embedding')
	parser.add_argument('--Generations', default=50, 
						help='number of generations to simulate')
	parser.add_argument('--M', default=500, 
						help='Population of each generation')
	parser.add_argument('--mutrate', default=5, 
						help='number of nodes to mutate per run')
	parser.add_argument('--k', default=1., 
						help='spring constant for edges')
	parser.add_argument('--d', default=10., 
						help='natural spring length')

#------------------Plotting Parameters----------------------------
	parser.add_argument('--plotting', default=True, 
						help='Used to suppress plotting functionality')
#------------------Saving Parameters----------------------------
	parser.add_argument('--saving', default=False, 
						help='Used to save positions file')
	parser.add_argument('--pathname', default='./', 
						help='filename for saving adjacency matrix as txt')
	parser.add_argument('--filename', default='Positions.txt', 
						help='filename for saving adjacency matrix as txt')
	args = parser.parse_args()
	return args         


def process_input(args):
# An adjacency matrix must be supplied
	A = np.genfromtxt(args.Adj, skip_header=1, delimiter=',')
	if args.Backbone:
		B = np.genfromtxt(args.Backbone, skip_header=1, delimiter=',')
	else:
		B = None
	if not args.Positions:
		X = None
	else:
		X = np.loadtxt(args.Positions)
	N = np.shape(A)[0]
	savefile = "".join([str(args.pathname), str(args.filename)])
	return N, A, B, X, int(args.Generations), int(args.M), int(args.mutrate), float(args.k), int(args.d), savefile, args.plotting, args.saving


def float_to_bin(num):
	return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)


def bin_to_float(binary):
	return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

 
def Generate_Parents(N, M, scale=10):
	X = np.zeros((M, N, 3))
	for m in range(M):
		X[m, :, :] = scale*(np.random.rand(N,3)-0.5)
	return X


def Mate(X,Y):
	Z = np.zeros_like(X)
	for n in range(np.shape(X)[0]):
		xbin = [float_to_bin(X[n,i]) for i in range(3)]
		ybin = [float_to_bin(Y[n,i]) for i in range(3)]
		cut = np.random.randint(32, size=3)
		Z[n,:] = [bin_to_float(''.join([xbin[i][:cut[i]], ybin[i][cut[i]:]])) for i in range(3)]	
	return Z

def Find_Energy(inputs):
	G, X, k, d = inputs
	Energy = 0
	for e in G.edges():
		Energy += 0.5 * k * (np.linalg.norm(X[e[0],:]-X[e[1],:]))**2
	for i in range(G.order()):
		for j in range(i+1,G.order()):
			Energy += 1./np.linalg.norm(X[i,:]-X[j,:])
	return Energy

def SeparateEm(N, A, Generations, M, k, d, cores=None):
	if not cores:
		cores = cpu_count()
	G = nx.Graph(A)
	# Use a GA to get a reasonable starting point
	MM = int(M/3)			# 1/3 best 1/3 mate 1/3 mutate
	X = Generate_Parents(N,M,scale=d)
	for g in range(Generations):
		pool = Pool(cores)
		results = pool.map(Find_Energy, [(G,X[m, :, :], k, d) for m in range(M)])	
		pool.close()
		Es = [i for i in results]
		best = np.argsort(Es)
		# keep the best half
		X[:MM:,:] = X[best[:MM],:,:]
		# if g % 10 == 0:
		# 	print(g,Es[best[0]])
		cnt = MM
		# mate the best half
		parents = np.random.choice(range(MM), size=(MM,2))
		for m in range(MM):
			# mate
#			parents = np.random.choice(range(MM), size=2)
			X[cnt,:,:] = Mate(X[parents[m,0],:,:], X[parents[m,1],:,:])
			# mutate
			mutrate = np.random.randint(N/2)
			whichones = np.random.randint(N, size=mutrate)
			X[MM+cnt,:,:] = X[np.random.randint(MM),:,:]
			X[MM+cnt,whichones,:] += np.random.normal(0,d,size=(mutrate,3))
			cnt+=1 
	# Use physics to make better
	X_best = X[0,:,:]
	T = 500
	for t in range(1,T):
		F = np.zeros((N,3))
		for x in range(N):
			for y in G.neighbors(x):
				F[x,:] += k* (np.linalg.norm(X_best[x,:]-X_best[y,:]))*(X_best[y,:]-X_best[x,:])/np.linalg.norm(X_best[x,:]-X_best[y,:])
			for y in range(N):
				if x!=y:
					F[x,:] += (X_best[x,:]-X_best[y,:])/np.linalg.norm(X_best[x,:]-X_best[y,:])**3
		X_best += 0.1*(0.1+(T-t)/T)*F
#	print(Find_Energy((G, X_best, k, d)))
	return X_best

def Plot3D(G,X,width=0.5, dark=False, color=None, axis=None):
	if dark:
		plt.style.use('dark_background')
		if not color:
			color = 'w'
	else:
		plt.style.use('default')		
		if not color:
			color = 'k'
	if axis:
		ax = axis
		fig = plt.gcf()
	else:
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
	for e in G.edges():
		ax.plot([X[e[0],0],X[e[1],0]], 
				[X[e[0],1],X[e[1],1]], 
				[X[e[0],2],X[e[1],2]],color, linewidth=width)
	if axis is None:
		ax.scatter(X[:,0], X[:,1], X[:,2], '.', s=75)
		ax.grid(False)
		plt.axis('off')	
	ax.view_init(elev=22, azim=146)
	ax.dist=8

	return fig, ax 

def Plot2D(G,X=None,width=0.5, dark=False, color=None, axis=None):
	if dark:
		plt.style.use('dark_background')
		if not color:
			color = 'w'
	else:
		plt.style.use('default')		
		if not color:
			color = 'k'
	if axis:
		ax = axis
		fig = plt.gcf()
	else:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	# if not X:
	# 	C = nx.spring_layout(G,scale=2.0)
	# 	X = np.array([[C[i][0],C[i][1]] for i in G.nodes()])
	# 	print(X)
	for e in G.edges():
		print(e)
		ax.plot([X[e[0],0],X[e[1],0]], 
				[X[e[0],1],X[e[1],1]], 
				color, linewidth=width)
	if axis is None:
		ax.scatter(X[:,0], X[:,1], s=400, marker='.')
		ax.grid(False)
		plt.axis('off')	
	return fig, ax 

def fig2img(fig):
	"""Convert a Matplotlib figure to a PIL Image and return it"""
	buf = io.BytesIO()
	fig.savefig(buf)
	buf.seek(0)
	img = Image.open(buf)
	return img

def MakePlot3DBackbone(G_name, N, G,X, ordered_edges, dark=True):
	plt.close()
	fig, ax = Plot3D(G,X,width=1.0, dark=dark)
	ax.view_init(0, 0)
	ax.dist=8
	images= [fig2img(fig)]
	plt.close()
	az, el = 0, 0
	for i in range(10):
		fig, ax = Plot3D(G,X,width=1.-0.05*i, dark=dark)
		ax.view_init(elev=el, azim=az)
		ax.dist=8
		images. append(fig2img(fig))
		plt.close()
		az += 1
		el += 1
		if el>270:
			el-=360
	if N<=100:
		for e in ordered_edges:
			e = [int(e[0]),int(e[1])]
			G.remove_edge(e[0], e[1])
	#		nx.draw(G, pos=Positions)
			fig, ax = Plot3D(G,X, width=1.0, dark=dark)
			ax.plot([X[e[0],0],X[e[1],0]],
					[X[e[0],1],X[e[1],1]],
					[X[e[0],2],X[e[1],2]], 'r',linewidth=0.75)
			ax.view_init(elev=el, azim=az)
			ax.dist=8
			az += 1
			el += 1
			if el>270:
				el-=360
			images. append(fig2img(fig))
			plt.close()
	else:
		G.remove_edges_from(ordered_edges)
		for i in range(20):
			fig, ax = Plot3D(G,X, width=0.5+0.025*i, dark=dark)
			for e in ordered_edges:
				e = [int(e[0]),int(e[1])]
				ax.plot([X[e[0],0],X[e[1],0]],
						[X[e[0],1],X[e[1],1]],
						[X[e[0],2],X[e[1],2]], 'r', linewidth=0.025*(20-i))
			ax.view_init(elev=el, azim=az)
			ax.dist=8
			images. append(fig2img(fig))
			plt.close()
			az += 1
			if el>270:
				el-=360
			el += 1
	for i in range(50):
		fig, ax = Plot3D(G,X, width=1.0, dark=dark)
		ax.view_init(elev=el, azim=az)
		ax.dist=8
		images. append(fig2img(fig))
		plt.close()
		az += 2
		el += 2
		if el>270:
			el-=360
	images[0].save("".join(['./BackboneGIF',str(G_name),'.gif']), save_all=True, 
					append_images=images[1:], 
					optimize=False, duration=100, loop=0)
	return


if __name__=='__main__':

	args = get_input()
	N, A, B, X, Generations, M, mutrate, k, d, savefile, plotting, saving = process_input(args)
	if X is None:
		X = SeparateEm(N, A, Generations, M, k, d)	
#	dark = True
	dark = False
	print(np.shape(X))
	if plotting and np.shape(X)[1]==3:
		if B is None:
			G = nx.Graph(A)
			fig, ax = Plot3D(G,X,width=1.0, dark=dark)
			plt.show()
		elif plotting=='skeleton':
			G = nx.Graph(A)			
			H = nx.Graph(B)
			fig, ax = Plot3D(G,X,width=0.5, dark=dark, color='r')
			fig, ax = Plot3D(H,X,width=1.0, dark=dark, color=None, axis=ax)			
			plt.show()			
		else:
			H = nx.Graph(B)
			fig, ax = Plot3D(H,X,width=1.0, dark=dark)
			plt.show()			
	elif plotting and np.shape(X)[1]==2:
		if B is None:
			G = nx.Graph(A)
			fig, ax = Plot2D(G,X,width=1.0, dark=dark)
			plt.show()
		elif plotting=='skeleton':
			G = nx.Graph(A)			
			H = nx.Graph(B)
			fig, ax = Plot2D(G,X,width=0.5, dark=dark, color='--r')
			fig, ax = Plot2D(H,X,width=2.0, dark=dark, color=None, axis=ax)			
			plt.show()			
		else:
			H = nx.Graph(B)
			fig, ax = Plot2D(H,X,width=1.0, dark=dark)
			plt.show()			

	if saving:
		np.savetxt(savefile, X)