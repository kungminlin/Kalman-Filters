import numpy as np

class Kalman_Filter:
	"""
	Arguments:
	A -- State Transition Matrix
	x -- State Model ([('variable1': <initial_value_1>), ('variable2': <initial_value_2>), ...])
	Q -- Process Noise Co-Variance Matrix
	H -- Observation Matrix (['variable1', 'variable2', 'variable3', ...])
	R -- Measurement Noise Covariance Matrix ([('variable1': <noise1>), ('variable2': <noise2>), ...]) - Preset is 10 for all variables; Variables specified must match exactly with the variables in Observation Matrix H.
	"""
	def __init__(A, Q, x, H, R=None):
		self.A = A
		self.Q = Q
		self.states = x
		self.x = np.array([var[1] for var in x])
		self.H = np.zeros(shape=(len(H),len(self.x)))
		for i in range(len(H)):
			self.H[i][[var[0] for var in x].index(H[i])] = 1
		self.R = np.identity(len(H)) * 10
		for noise in R if R is not None:
			index = H.index(noise[0])
			self.R[index][index] = noise[1]
		self.P = np.zeroes(shape=(len(x),len(x)))

	def update(new_state):
		self.x = self.A.dot(self.x)
		self.P = self.A.dot(self.P).dot(self.A.T) + self.Q
		y = new_state - self.H.dot(x) 							# Innovation Factor
		S = self.H.dot(self.P).dot(self.H.T) + self.R 							# Innovation Co-Variance
		K = self.P.dot(self.H.T).dot(np.linalg.pinv(S))				# Kalman Gain
		self.x = self.x + K.dot(y)
		self.P = (np.identity(len(self.x)) - (K.dot(self.H))).dot(self.P)

	def get_state():
		return self.x


