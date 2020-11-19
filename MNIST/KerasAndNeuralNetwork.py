import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train_softmax = np.zeros(shape=(60000, 10), dtype=np.float32)
y_test_softmax = np.zeros(shape=(10000, 10), dtype=np.float32)

for i in range(10):
    y_train_softmax[np.where(y_train == i), i] = 1
    y_test_softmax[np.where(y_test == i), i] = 1

y_train_softmax = tf.convert_to_tensor(y_train_softmax)
y_test_softmax = tf.convert_to_tensor(y_test_softmax)

x_train = x_train/255
x_test = x_test/255

# Input of the model is flatten Keras takes layers. Two layer machine learning can be done by iterating one layer
# In mathematical terms applying same function over and over
# Function in the hidden layer has to be non-linear function, as combination of two linear functions is lin. function
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name="MyFlatten"),
    # Hidden intermediate layer, parameters can be changed on demand in the middle layer, 20 Neurons
    # Look up relu function
    tf.keras.layers.Dense(20, activation="relu", name="MyDense"),
    # Output layer
    tf.keras.layers.Dense(10, activation="softmax", name="MyDense2")],
    name="NumberDetection"
)

# Build the model and check if everythin is alright
model.build(input_shape=(60000, 28, 28))
model.summary()

# Compile the model
model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["accuracy"])

# Runs the model
model.fit(x_train, y_train_softmax, epochs=20)

# Look up what an epoch is. Increases and decreases delta/ learning rate in epochs by itself (cycles of running)

# Ask models to predict output
predictions = model.predict(x_test)

# Test model, probably some overfitting here.
print(tf.reduce_sum(
    tf.cast(
        tf.argmax(
            predictions, axis=1) == y_test,
        tf.int16)).numpy()
)

# If accuracy on learning set is too great (bigger than test set) there is overfitting occuring
# Understanding too good how train set works
# Neural networks are not learning, they are just fitting (No generalization)

# Problem is interpolation and exterpolation problem for fit. The more fitting parameter one has the worse the
# fitting will be between points for interpolation.

# For exterpolation outside of the boundary of validity guess how it will continue. Polynomial with quadratic numbers
# is not good for fitting

# Deep learning; means multilayer neural network
