 tensor: a generalization of vectors and matrices to potentially higher dimensions
 ou a collection of numbers which is arranged into a particular shape
 
 import tensorflow as tf
 
 # 0D Tensor
 d0 = tf.ones((1,))
 
 # 1D Tensor
 d1 = tf.ones((2,))
 
 # 2D Tensor
 d2 = tf.ones((2, 2))
 
 # 3D Tensor
 d3 = tf.ones((2, 2, 2))
 
 print(d3.numpy())
 
 constant is the simplest category of tensor. It cant be changed and it cannot be trained (not trainable). can have any dimension 
 
 from tensorflow import constant 
 
 # Define a 2x3 constant
 a = constant(3, shape = [2, 3]) # 2x3 tensor of 3's
 
 # Define a 2x2 contant
 b = constant([1, 2, 3, 4], shape = [2, 2])
 
 # Define a variable
 a = tf.Variable([1, 2, 3, 4, 5, 6], dtype = tf.float32) # or tf.int16
 # Define a constant
 b = tf.constant(2, tf.float32)
 # Compute their product
 c = tf.nultiply(a, b) # or simply a * b
 
 add()  operations performs element-wise addition with two tensors. Element-wise addition requires both tensors to have the same shape
 element-wise multiplication: multiply()
 matrix multiplication: matmul()
 
 # Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)

# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))


# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)

# Compute and print the error
error = bill - billpred
print(error.numpy())




# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (28*28, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (28*28*3, 1))


def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x, x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))



You are given a black-and-white image of a letter, which has been encoded as a tensor, letter. You want to determine whether the letter is an X or a K. You don't have a trained neural network, but you do have a simple model, model, which can be used to classify letter.

The 3x3 tensor, letter, and the 1x3 tensor, model, are available in the Python shell. You can determine whether letter is a K by multiplying letter by model, summing over the result, and then checking if it is equal to 1. As with more complicated models, such as neural networks, model is a collection of weights, arranged in a tensor.

# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output, 0)
print(prediction.numpy())



# PART2
# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing['price'])



# Import numpy and tensorflow with their standard aliases
import numpy as np
import tensorflow as tf

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)



# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())

