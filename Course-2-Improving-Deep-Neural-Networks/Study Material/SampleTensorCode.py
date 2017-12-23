import tensorflow as tf

# Defining Constants : Value doesn't change. 
# It's constant. 
# No input. No output.
# constObj = tf.constant(value, [Datatype])

node1 = tf.constant(3.0, tf.float32)

# Creating a session

with tf.Session() as session :
	print(session.run(node1))

# OR

session = tf.Session()	
print(session.run(node1))
session.close()


# Conputation Graph : Involves methods defined in tensorflow.

a = tf.constant(5.0)
b = tf.constant(6.0)

c = a * b

with tf.Session() as session :
	print(session.run(c))

# Defining Placeholder : Graph can be parameterized. 
# Placeholder is promise to provide value later.
# Assumes imput from feed_dict parameter.
# placeholderVar = tf.placeholder(datatype)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

with tf.Session() as session :
	print(session.run(adder_node, feed_dict = {a:[1, 3], b:[2, 4]}))

# Variables : To make model trainable, we need to be able to modify graph.
# New output for same input.
# Allow us to add trainable parameters to graph.
# Need to be explicitly initialized.
# variableVar = tf.Variable(value, datatype)

# For initalisation one may use following :

W1 = tf.get_variable("W1", shape, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1", shape, initializer = tf.zeros_initializer())

# Example :

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b

squared_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_delta)

init = tf.global_variables_initializer()

# Usage of optimizers : Check change in loss wrt change in variable & Update accordingly
optimizer = tf.train.GradientDescentOptimizer(0.01) # learning rate
train = optimizer.minimize(loss)

with tf.Session() as session :
	session.run(init)
	for i in range(1000) :
		sesssion.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
	print(session.run([W, b]))	
	print(session.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))



# one_hot Method :
# Creates a matrix where the i-th row corresponds to the ith class number and the jth column
# corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
# will be 1. 

one_hot_matrix = tf.one_hot(labels, depth, axis) 	

# Initialize a vector of Zeros and Ones :

ones = tf.ones(shape)
    
with tf.Session() as sess :
    ones = sess.run(ones)

# Similarly

zeros = tf.zeros(shape)   
with tf.Session() as sess :
    zeros = sess.run(zeros) 

# Other important Methods :

tf.add(...,...) to do an addition
tf.matmul(...,...) to do a matrix multiplication
tf.nn.relu(...) to apply the ReLU activation
tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))

tf.placeholder(tf.float32, name = "...")
tf.sigmoid(...)
tf.softmax(...)
sess.run(..., feed_dict = {x: z})









