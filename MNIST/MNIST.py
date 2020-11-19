import tensorflow as tf
import numpy as np
# Good way to avoid overfitting of neural network; split database into 80-20% and then use the 20% to test model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.cast(x_train.reshape(60000, 784), dtype=np.float32)    # Train network
x_test = tf.cast(x_test.reshape(10000, 784), dtype=np.float32)  # Test network

# Can be done with own data, but is very inefficient

y_train_softmax = np.zeros(shape=(60000, 10), dtype=np.float32)
y_test_softmax = np.zeros(shape=(10000, 10), dtype=np.float32)

# Picture is set to one on a ten vector, where it is the number
# Numpy vectors are mutable, but tf vectors (like tf zeros) are not

for i in range(10):
    y_train_softmax[np.where(y_train == i), i] = 1
    y_test_softmax[np.where(y_test == i), i] = 1

y_train_softmax = tf.convert_to_tensor(y_train_softmax) # Not mutable anymore
y_test_softmax = tf.convert_to_tensor(y_test_softmax)   # Work like tuples


def setup_loss_function(
        X_train,
        Y_train,
        batch_size=5000):
    W = tf.Variable(initial_value=tf.ones(shape=(784, 10)))
    B = tf.Variable(initial_value=tf.ones(shape=(10)))

    # Repeat over batches in order to get all global minima (not local fake ones as they change dramatically) in batches
    # True minimum should not depend on the training pictures, should always find correct one
    dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, Y_train)
    ).batch(batch_size).repeat(5000)
    batches = iter(dataset)

    def loss():
        X, Y = next(batches)
        return tf.reduce_sum(
            (tf.nn.softmax(tf.matmul(X, W)+B)-Y)**2
        )

    return loss, [W, B]

# Normalize the input as to not take too much computing power
x_test = x_test/255
x_train = x_train/255

# Use optimizer for computing minimum
opt = tf.keras.optimizers.Adadelta(
    learning_rate=0.13,
    epsilon=0.0005
)

# Input variables in loss function
loss, variables = setup_loss_function(x_train, y_train_softmax)


for i in range(5000):
    opt.minimize(loss, variables)
    if i % 50 == 0:
        print("Loss function at iteration", i, ":", loss().numpy())

# Change options like batch size as well as learning rate to optimize process
# Depends on the initial parameters, almost on all variables

fitted_W, fitted_B = variables
print(fitted_W, fitted_B)

# True is one and false is zero
# tf.argmax gives the index of the maximal entry
print(tf.reduce_sum(
    tf.cast(
        tf.argmax(
            tf.nn.softmax(
                tf.matmul(x_test, fitted_W)
                +fitted_B),
            1)==y_test,
        tf.int16
    )
))

# It's all about finding the best function or finding the best model
# Only one layer as softmax is applied only once. Softmax of softmax on L_i can be used -> multiples layers