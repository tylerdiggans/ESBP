import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint, solve_ivp
from copy import deepcopy

def MSFLiapunov_ll(inputs):
	s, dt, DFunc, K, DH, params = inputs
	# This is the main lyapunov exponent estimator function for the 
	#			MSF Analysis
	# INPUTS:
	# 	s     - A synchronous orbit on attractor of the basic system 
	# 	dt 	  - time step for orbit needed for calculation of LLE
	# 	DFunc - The T=dt tangent map for the network from MSF 
	#   K     - normalized Coupling strength for MSF
	#   DH    - tangent map for the coupling matrix delta_ij
	#   params - the parameters for the basic chaotic system
	#
	# OUTPUT:
	#	Largest Lyapunov Exponent Estimation 
	#
	n,T = np.shape(s)		# Iterations for LLE estimation (points on orbit)
	q = np.identity(n)		# Initial condition for QR process
	LE = np.zeros((n,))		# Lyapunov exponent estimation initialized 
	for t in range(T):
		# Evaluate the tangent map at the syncrhonous orbit points
		#     and right multiply by the previous iterations Q matrix
		Tk = np.dot(DFunc(s[:,t], dt, K, DH, params),q)  # New ball
		q, r = np.linalg.qr(Tk)
		# The LE estimators are obtained from the R matrix diagonal elements
		LE += np.log(abs(np.diag(r)))/dt
	return K, LE[0]/T

def MSF_Liapunov(s, dt, DFunc, K, DH, params):
	# This is the main lyapunov exponent estimator function for the 
	#			MSF Analysis
	# INPUTS:
	# 	s     - A synchronous orbit on attractor of the basic system 
	# 	dt 	  - time step for orbit needed for calculation of LLE
	# 	DFunc - The T=dt tangent map for the network from MSF 
	#   K     - normalized Coupling strength for MSF
	#   DH    - tangent map for the coupling matrix delta_ij
	#   params - the parameters for the basic chaotic system
	#
	# OUTPUT:
	#	Largest Lyapunov Exponent Estimation 
	#
	n, T = np.shape(s)		# Iterations for LLE estimation (points on orbit)
	q = np.identity(n)		# Initial condition for QR process
	LE = np.zeros((n,))		# Lyapunov exponent estimation initialized 
	for t in range(T):
		# Evaluate the tangent map at the syncrhonous orbit points
		#     and right multiply by the previous iterations Q matrix
		Tk = np.dot(DFunc(s[:,t], dt, K, DH, params),q)  # New ball
		q, r = np.linalg.qr(Tk)
		# The LE estimators are obtained from the R matrix diagonal elements
		LE += np.log(abs(np.diag(r)))/dt
	return LE[0]/T


def MSF(Func, DFunc, H, params, plotting):
	# This is the main lyapunov exponent estimator function for the 
	#			MSF Analysis
	# INPUTS:
	# 	Func  - The chaotic map for the basic system from MSF 
	# 	DFunc - The T=dt tangent map for the network from MSF
	#	H     - Coupling matrix  e.g. delta_ij 
	#   params- the parameters for the basic chaotic system
	#	plotting - options for different plots defaults to histogram
	#
	# OUTPUT:
	#	An interval of normalized coupling strengths K for which 
	# 		synchronization is possible 
	#

	# Simulate to obtain a point near attractor as synchronous orbit IC
	d = np.shape(H)[0]
	y0 = np.random.rand(d)
	sol = solve_ivp(Func, [0,5000], y0, t_eval=[5000], 
					method='RK45', args=params)

	# uncomment if you want to see the attractor
	if plotting=='Attractor':
		solforplot = solve_ivp(Func, [0,5000], y0, range(5000), 
								method='RK45', args=params)
		x = solforplot.y.transpose()
		fig =plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(x[:,0], x[:,1], x[:,2])
		plt.show()

	T, dt = [50, 0.01]			# Total time and time step for LLE estimation
	# Increment the synchronous state s seperately for Liapunov
	sol = solve_ivp(Func, [0, T], sol.y.reshape((d,)), 
					t_eval=np.linspace(0,T,num=int(T/dt)), 
					method='RK45', args=params)
	# Now Use Custom Bisection Algorithm 
	# to find the zeros of the MSF 
	# 	(assumes 2, but searches until maxK)
	maxK, dK = [20., 0.1]	# largest allowable K and Kstep size
	Kb = deepcopy(dK) 		# initialize for while loop criteria
	zeroses = []			# collect al zeros
	cnt = 0
	while (Kb<maxK): # and len(zeroses)<2: 
		# Initialize interval for zero searching
		if len(zeroses)<1:
			Ka = 0
		else:
			Ka = zeroses[-1]+0.001
			Kb = Ka+dK
		La = MSF_Liapunov(sol.y, dt, DFunc, Ka, H, params)	
		Lb = MSF_Liapunov(sol.y, dt, DFunc, Kb, H, params)	
		# Psi[cnt] = deepcopy(Lb)
		cnt += 1
		while (La*Lb>0) and (Kb<maxK):
			Ka = deepcopy(Kb)
			Kb += dK
			La = deepcopy(Lb)	
			Lb = MSF_Liapunov(sol.y, dt, DFunc, Kb, H, params)	
			# Psi[cnt] = deepcopy(Lb)
			cnt += 1
		# Change the number of iterations to increase accuracy
		for i in range(12):
			Kc = (Ka+Kb)/2.
			Lc = MSF_Liapunov(sol.y, dt, DFunc, Kc, H, params)	
			if La*Lc>0:
				Ka = deepcopy(Kc)
				La = deepcopy(Lc)
			elif La*Lc==0.:
				zeroses.append(Kc)
			else:
				Kb = deepcopy(Kc)
				Lb = deepcopy(Lc) 
		zeroses.append(Kc)
	print(zeroses)
	return zeroses


def PlotMSF(Func, DFunc, H, params, cores):
	# This is the main lyapunov exponent estimator function for the 
	#			MSF Analysis
	# INPUTS:
	# 	Func  - The chaotic map for the basic system from MSF 
	# 	DFunc - The T=dt tangent map for the network from MSF
	#	H     - Coupling matrix  e.g. delta_ij 
	#   params- the parameters for the basic chaotic system
	#	plotting - options for different plots defaults to histogram
	#
	# OUTPUT:
	#	An interval of normalized coupling strengths K for which 
	# 		synchronization is possible 
	#

	# Simulate to obtain a point near attractor as synchronous orbit IC
	if isinstance(H, float):
		n = 1
	else:
		n = np.shape(H)[0]
	y0 = np.random.rand(n)
	sol = solve_ivp(Func, [0,5000], y0, t_eval=[5000], method='RK45', args=params)
	s = sol.y.reshape((n,))		# a synchronous orbit IC
	# Accuracy of LLE estimator
	T, dt = [50, 0.0025]			# Total time and time step for LLE estimation
	N = int(T/dt)				# Numer of iterations used for LLE estimation
	# Increment the synchronous state s seperately for Liapunov
	sol = solve_ivp(Func, [0, T], s, t_eval=np.linspace(0,T,num=N), method='RK45', args=params)
	# Now Use Custom Bisection Algorithm 
	# to find the zeros of the MSF 
	# 	(assumes 2, but searches until maxK)
	maxK, dK = [20., 0.1]	# largest allowable K and Kstep size
	K = int(maxK/dK)
	Ks = np.linspace(0, maxK-dK, num=K)
	pool = Pool(cores)
	results = pool.map(MSFLiapunov_ll, [(sol.y, dt, DFunc, Ks[i], H, params) for i in range(K)])	
	pool.close()
	Psi = np.array([[k,i] for k,i in results])
	Psi = Psi[Psi[:,0].argsort()]
	zeroses = []
	for k in range(K-1):
		if Psi[k,1]*Psi[k+1,1]<0:
			Ka, Kb = Psi[k:k+2,0]
			La, Lb = Psi[k:k+2,1]
			for i in range(12):
				Kc = (Ka+Kb)/2.
				Lc = MSF_Liapunov(sol.y, dt, DFunc, Kc, H, params)	
				if La*Lc>0:
					Ka = deepcopy(Kc)
					La = deepcopy(Lc)
				elif La*Lc==0.:
					zeroses.append(Kc)
				else:
					Kb = deepcopy(Kc)
					Lb = deepcopy(Lc) 
			zeroses.append(Kc)
	fig,ax = plt.subplots(1)
	ax.plot(Ks,Psi[:,1])
	ax.plot(Ks,np.zeros_like(Ks), 'k--')
	ax.text(2,2, '%s\n' % zeroses) 
	ax.set_xlabel('K')
	ax.set_ylabel('$\Psi(K)$ (MSF)')
	ax.set_title('Master Stability Function \n for Harmonic Oscillator with Damper')
	plt.show()
	print(zeroses)
	return zeroses
