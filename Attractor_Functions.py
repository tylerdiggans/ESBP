import numpy as np

#####################################################################
###   This is a set of function used for synchronization of  
###	 chaotic oscillators and Master Stability Function Analysis
###	 For each system, there is a normal version, a network version
###	 and a function for the tangent map for estimation of the LLE	
###					 	
#####################################################################

def rossler(t,w,a,b,c):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Equations
	return [-w[1]-w[2], w[0] + a * w[1], b + w[2] * (w[0]-c)]

def rossler_ll(t,w,a,b,c):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Variables
	x = w[::3]  
	y = w[1::3]   
	z = w[2::3]
# Equations
	dwdt = np.zeros((len(w),))
	dwdt[::3] = -y-z
	dwdt[1::3] = x + a * y
	dwdt[2::3] = b + z * (x-c)
	return dwdt

def rossler_net(t,w,a,b,c,eLH):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Equations
	# Negative Laplacian connection matrix as in Pecora Carroll
	dwdt = np.array([-w[1]-w[2], 
			w[0] + a * w[1],
			b + w[2] * (w[0]-c)]) - np.dot(eLH,w)
	return dwdt 

def rossler_net_ll(t,w,a,b,c,eLH):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Variables
	x = w[::3]  
	y = w[1::3]   
	z = w[2::3]
# Equations
	# Negative Laplacian connection matrix as in Pecora Carroll
	dwdt = np.zeros((len(w),))
	dwdt[::3] = -y-z
	dwdt[1::3] = x + a * y
	dwdt[2::3] = b + z * (x-c)
	dwdt -= np.dot(eLH,w)
	return dwdt 

def DF_rossler(s, dt, K, DH, params):
	# This is the \tilde{DF}(s) for the params=(a,b,c) Rossler 
	# 		Equations with the normalized coupling constant K 
	#		and coupling matrix H from MSF papers
	#
	# 	i.e. dQ/dt = [DF(s) - KDH(s)] Q(t) 

	# s   -  a point on the synchronization trajectory
	# dt  -  the time step size for the trajectory orbit
	a, b, c = params
	DF = np.identity(3) + dt * (np.array([[0., -1., -1.],
			[1., a, 0.],
			[s[2], 0., s[0]-c]]) - K * DH)
	# we reshape the output for plugging into odeint again
	return DF


def lorenz(t,w,sig,rho,beta):
# Parameters for the Lorenz system
#    sig = 10.
#    rho = 28.
#    beta = 8./3.
# Equations
	return [sig*(w[1]-w[0]), w[0]*(rho-w[2])-w[1], w[0]*w[1]-beta*w[2]]

def lorenz_ll(t,w,sig,rho,beta):
# Parameters for the Lorenz system
#    sig = 10.
#    rho = 28.
#    beta = 8./3.
# Variables
	x = w[::3]  
	y = w[1::3]   
	z = w[2::3]
# Equations
	dwdt = np.zeros((len(w),))
	dwdt[::3] = sig * (y - x)
	dwdt[1::3] = x * (rho - z) - y
	dwdt[2::3] = x * y - beta * z
	return dwdt

def lorenz_net(t,w,sig,rho,beta,eLH):
# Parameters for the Lorenz system
#    sig = 10.
#    rho = 28.
#    beta = 8./3.
# Equations
	dwdt = np.array([sig * (w[1] - w[0]),
					w[0] * (rho - w[2]) - w[1], 
					w[0] * w[1] - beta * w[2]]) - np.dot(eLH,w)
	return dwdt 

def lorenz_net_ll(t,w,sig,rho,beta,eLH):
# Parameters for the Lorenz system
#    sig = 10.
#    rho = 28.
#    beta = 8./3.
# Variables
	x = w[::3]  
	y = w[1::3]   
	z = w[2::3]
# Equations
	dwdt = np.zeros((len(w),))
	dwdt[::3] = sig * (y - x)
	dwdt[1::3] = x * (rho - z) - y
	dwdt[2::3] = x * y - beta * z
	dwdt -= np.dot(eLH,w)
	return dwdt 

def DF_lorenz(s, dt, K, DH, params):
	# This is the \tilde{DF}(s) for the params=(a,b,c) Rossler 
	# 		Equations with the normalized coupling constant K 
	#		and coupling matrix H from MSF papers
	#
	# 	i.e. dQ/dt = [DF(s) - KDH(s)] Q(t) 

	# s   -  a point on the synchronization trajectory
	# dt  -  the time step size for the trajectory orbit
	sig, rho, beta = params
	DF = np.identity(3) + dt * (np.array([[-sig, sig, 0.],
			[rho-s[2], -1., -s[0]],
			[s[1], s[0], -beta]]) - K * DH)
	# we reshape the output for plugging into odeint again
	return DF

def chens(t,w,a,c,beta):
# Parameters for the Lorenz system
#    sig = 10.
#    rho = 28.
#    beta = 8./3.
# Equations
	return [a*(w[1]-w[0]), w[0]*(c-a-w[2])+c*w[1], w[0]*w[1]-beta*w[2]]

def chens_ll(t,w,a,c,beta):
# Parameters for the Chens system
#    a = 35.
#    c = 28.
#    beta = 8./3.
# Variables
	x = w[::3]  
	y = w[1::3]   
	z = w[2::3]
# Equations
	dwdt = np.zeros((len(w),))
	dwdt[::3] = a * (y - x)
	dwdt[1::3] = x * (c - a - z) + c * y
	dwdt[2::3] = x * y - beta * z
	return dwdt

def chens_net(t,w,a,c,beta,eLH):
# Parameters for the Chens system
#    a = 35.
#    c = 28.
#    beta = 8./3.
# Equations
	dwdt = np.array([a * (w[1] - w[0]),
					w[0] * (c - a - w[2]) + c * w[1], 
					w[0] * w[1] - beta * w[2]]) - np.dot(eLH,w)
	return dwdt 

def chens_net_ll(t,w,sig,rho,beta,eLH):
# Parameters for the Chens system
#    a = 35.
#    c = 28.
#    beta = 8./3.
# Variables
	x = w[::3]  
	y = w[1::3]   
	z = w[2::3]
# Equations
	dwdt = np.zeros((len(w),))
	dwdt[::3] = a * (y - x)
	dwdt[1::3] = x * (c - a - z) + c * y
	dwdt[2::3] = x * y - beta * z
	dwdt -= np.dot(eLH,w)
	return dwdt 

def DF_chens(s, dt, K, DH, params):
	# This is the \tilde{DF}(s) for the params=(a,b,c) Rossler 
	# 		Equations with the normalized coupling constant K 
	#		and coupling matrix H from MSF papers
	#
	# 	i.e. dQ/dt = [DF(s) - KDH(s)] Q(t) 

	# s   -  a point on the synchronization trajectory
	# dt  -  the time step size for the trajectory orbit
	a, c, beta = params
	DF = np.identity(3) + dt * (np.array([[-a, a, 0.],
			[c - a - s[2], c, -s[0]],
			[s[1], s[0], -beta]]) - K * DH)
	# we reshape the output for plugging into odeint again
	return DF

def harm(t,w,a,b,c):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Equations
	return [w[1], -w[0]]

def harm_ll(t,w,a,b,c):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Variables
	x = w[::3]  
	y = w[1::3]   
# Equations
	dwdt = np.zeros((len(w),))
	dwdt[::3] = y
	dwdt[1::3] = -x
	return dwdt

def harm_net(t,w,a,b,c,eLH):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Equations
	# Negative Laplacian connection matrix as in Pecora Carroll
	dwdt = np.array([w[1], 
			w[0]]) - np.dot(eLH,w)
	return dwdt 

def harm_net_ll(t,w,a,b,c,eLH):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Variables
	x = w[::3]  
	y = w[1::3]   
# Equations
	# Negative Laplacian connection matrix as in Pecora Carroll
	dwdt = np.zeros((len(w),))
	dwdt[::3] = y
	dwdt[1::3] = -x
	dwdt -= np.dot(eLH,w)
	return dwdt 

def DF_harm(s, dt, K, DH, params):
	# This is the \tilde{DF}(s) for the params=(a,b,c) Rossler 
	# 		Equations with the normalized coupling constant K 
	#		and coupling matrix H from MSF papers
	#
	# 	i.e. dQ/dt = [DF(s) - KDH(s)] Q(t) 

	# s   -  a point on the synchronization trajectory
	# dt  -  the time step size for the trajectory orbit
	a, b, c = params
	n = np.shape(s)[0]
	DF = np.identity(n) + dt * (np.array([[0., 1.],
			[-1., 0.]]) - K * DH)
	# we reshape the output for plugging into odeint again
	return DF

def simple(t,w,a,b,c):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Equations
	return [a]

def simple_ll(t,w,a,b,c):
# Parameters for the Rossler system
#    a = omega
# Equations
	dwdt = a * np.ones((len(w),))
	return dwdt

def simple_net(t,w,a,b,c,eLH):
# Parameters for the Rossler system
#    a = 0.2
#    b = 0.2
#    c = 7;
# Equations
	# Negative Laplacian connection matrix as in Pecora Carroll
	dwdt = np.array([w[1], 
			w[0]]) - np.dot(eLH,w)
	return dwdt 

def simple_net_ll(t,w,a,b,c,eLH):
# Parameters for the Rossler system
#    a = omega
# Equations
	# Negative Laplacian connection matrix as in Pecora Carroll
	dwdt = a * np.ones((len(w),))
	dwdt -= np.dot(eLH,w)
	return dwdt 

def DF_simple(s, dt, K, DH, params):
	# This is the \tilde{DF}(s) for the params=(a,b,c) Rossler 
	# 		Equations with the normalized coupling constant K 
	#		and coupling matrix H from MSF papers
	#
	# 	i.e. dQ/dt = [DF(s) - KDH(s)] Q(t) 

	# s   -  a point on the synchronization trajectory
	# dt  -  the time step size for the trajectory orbit
	a, b, c = params
	DF = 1. + dt * (-K * DH)
	# we reshape the output for plugging into odeint again
	return DF
