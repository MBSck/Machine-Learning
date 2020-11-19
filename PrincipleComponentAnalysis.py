import numpy as np
import matplotlib.pyplot as plt
# Try to solve the problem of overfitting
# Polynomial goes through all the points of dataset -> Has very larg fluctuations ->> Bad inter/-extrapolation of data

# Numpy can be used to calculate with matrizes immediately

# Find out variables that are maximally uncorrelated -> Keep this information and drop the correlated information
# -->> Not losing that much information (Some is always lost)

# Consider simple database (2D)
x = np.array([2, 2, 1, 4, 4, 4, 4, 2, 4, 4, 0])
y = np.array([5, 4, 2, 9, 8, 8, 9, 5, 8, 9, 1])

# Some points are scattered along a line plus fluctuations
"""
plt.scatter(x, y)
plt.plot(np.arange(0, 5), 2*np.arange(0, 5), color="red")
plt.show()
"""
# Principle Component analysis
# How to represent two dimensional database as one dimensional database
# Use distance of red line from origin to point as a new coordinate and
# disregard fluctuation as irrelevant/noise

# Disentangle Signal (Line) from the Noise
# Covariance is positive if both coords are above mean value or below
# -> Positive correlation between x & y

mean_x = np.mean(x)     # 1/N sum over all x
mean_y = np.mean(y)

# Covariance : 1/(N-1)sum((x-mean_x)*(y-mean_y))
# Positive correlation -> Positive value of x and positive value of y or negative of both -> Consistently the same
# Negative correlation -> Positive value of x and negative of y or the other way around -> Consistently the same
# No correlation -> Sometimes positive value of x and neg of y or the other way around -> Around zero

# print(sum((x-mean_x)*(y-mean_y)))

# If data aligned positive slope -> Positive correlations; alpha pos.
# If data aligned negative slope -> Negative correlations; alpha neg.

# With Sigma_XX, Sigma_YY and Sigma_XY can construct matrix

sigma = np.array([[sum((x-np.mean(x))**2)/10, sum((x-np.mean(x))*(y-np.mean(y)))/10],
                  [sum((x-np.mean(x))*(y-np.mean(y)))/10, sum((y-np.mean(y))**2)/10]])

# print(sigma)

# Mathematical theorem, All symmetric matrices can be decomposed -> Spectral decomposition
# With O as orthogonal matrix and Sigma = O Lambda O^T, where O*O^T = 1
# Lamda_ii = Lambda_i -> Where lambda_i are the eigenvalues
# Sigma * v_i = lambda_i * v_i

# -> O is rotation matrix and the rotation is performed twice
# Sigma is diagonalized by O

# Eigenvalues and eigenvectores
eigvl, O = np.linalg.eigh(sigma)
eigvc = O.T

# print(eigvl)
# print(eigvc)

# Verify that they are eigenvectors
# print(sigma@eigvc[0], eigvl[0]*eigvc[0])    # @ does the matrix multiplication, @ = __matmul__

# Vector is coordinate in two dimensional space -> Arrow that shows direction
"""
plt.scatter(x, y)
plt.plot(np.arange(-10, 10)*eigvc[0, 0], np.arange(-10, 10)*eigvc[0, 1])
plt.plot(np.arange(-10, 10)*eigvc[1, 0], np.arange(-10, 10)*eigvc[1, 1])
plt.show()
"""

# Shows two directions one where the data is aligned and one that is orthogonal
# To that where the data occurs, but matplotlib is stretching, so it doesnt look like orthogonal
# Reduce the noise direction, largest in magnitude eigenvalues is the direction where the relevant data is aligned

# PCA will work very well, if data is correlated and one value is much larger than the other

plt.scatter(x, y)
plt.plot(np.arange(-10, 10)*eigvc[1, 0], np.arange(-10, 10)*eigvc[1, 1])
plt.show()

# Sigma can be approximated very easily, by setting some eigenvalues to zero. -> Set the ones to zero that are smaller
# I.e. have smaller variance and contain less information.

# As this does not reference to the labels, it is not connected to the database, but to the most important valeus
# ---->>> Unsupervised learning

