import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Convert integers to floats
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

# Substract mean from database
mn = tf.reduce_mean(x_train, axis=0)    # axis=0 means that mn is not a number but a tensor, gets mean of every entry
# If axis would be not set then there would be mean over all numbers and not one for every picture
V = x_train - mn    # Subract mean

V = tf.reshape(V, shape=(60000, 28*28))     # Flatten the matrix

# As mean is removed already sigma simplifies
sigma = tf.matmul(tf.transpose(V), V)

# Eigenvalues are given in reverse order
eigs, O = tf.linalg.eigh(sigma)
eigv = tf.transpose(O)

# Singular value decomposition -> more stable algorithm
# Should all be positive, but maybe numerical error through float

eigv = tf.reverse(eigv, axis=[0])

for i in range(12):
    plt.imshow(
        eigv[i].numpy().reshape(28, 28),
        cmap="Greys"
    )
    plt.show()

# Projections of MNIST on to eigenvectors -> compressed picture
projections = tf.matmul(
    eigv[:50],
    tf.transpose(V)
)

# Get back the original pictures
projected_pictures = tf.matmul(
    tf.transpose(projections),
    eigv[:50]
)

projected_pictures = mn + tf.reshape(
    projected_pictures, (60000, 28, 28)
)

for i in range(8):
    plt.imshow(
        projected_pictures[i].numpy(),
        cmap="Greys"
    )
    plt.show()
    plt.imshow(
        x_train[i].numpy(),
        cmap="Greys"
    )
    plt.show()

# Every matrix can be written  a combination of coefficient and a base. Where the coefficients are the reduced data
# And a base can be for example a shifting one -> Stupid base
# PCA tries to find optimal base, so every picture can be written as a truncated sum up to a lower index, but
# retaining as much information as possible

# LECTURE 14

# Eigenvectors time eigenvectors results in orthogonal basis as you get one for first entry and 0 for following
# Eigenvectordecomposition -> orthonormal vector space -> All eigenvectors form 90 degree angle
