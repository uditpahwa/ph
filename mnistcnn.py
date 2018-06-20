import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/" , one_hot=True)
n_input = 784
n_out = 10
# hyper_parameters
learning_rate = 0.01
training_epochs = 2000
display_epoch = 10
batch_size = 128
keep_rate = 0.8

# place_holders
x = tf.placeholder('float',[None, n_input])
y = tf.placeholder('float',[None,n_out])

#defining the 2-dimentional convolution function with an image 'x' as input and weights is the filter 
def conv2d(x,weights):
	
	return tf.nn.conv2d(x,weights,strides = [1,1,1,1], padding='SAME')

#defining the max-pool function with an image 'x' as input
def max_pool(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides= [1,2,2,1],padding= 'SAME')
# model
## This Cnn consist of 2 conv layers and one fully-connected layer and one output layer
### I've used the relu activation function for the fully connected layer
def Convolutional_Neural_Network(x):

	weights = {'W_Conv1': tf.Variable(tf.random_normal([5,5,1,32])),
			   'W_Conv2': tf.Variable(tf.random_normal([5,5,32,64])),
			   'W_fc_layer': tf.Variable(tf.random_normal([7*7*64,1024])),
			   'Out': tf.Variable(tf.random_normal([1024,n_out]))}
	
	biases = {'B_Conv1': tf.Variable(tf.random_normal([32])),
			   'B_Conv2': tf.Variable(tf.random_normal([64])),
			   'B_fc_layer': tf.Variable(tf.random_normal([1024])),
			   'Out': tf.Variable(tf.random_normal([n_out]))}
	x = tf.reshape(x, [-1,28,28,1])

	layer1 = tf.add(conv2d(x,weights['W_Conv1']),biases['B_Conv1'])
	layer1 = max_pool(layer1)

	layer2 = tf.add(conv2d(layer1,weights['W_Conv2']),biases['B_Conv2'])
	layer2 = max_pool(layer2)

	layer3 = tf.reshape(layer2, [-1,7*7*64])

	layer3 = tf.add(tf.matmul(layer3,weights['W_fc_layer']),biases['B_fc_layer'])
	layer3 = tf.nn.relu(layer3)
	
	output = tf.add(tf.matmul(layer3,weights['Out']),biases['Out'])
	
	return output

# Training of the NN:

def training_model(x):
	prediction = Convolutional_Neural_Network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels= y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epochs in range(display_epoch):
			epoch_loss = 0

			for _ in range(training_epochs):
				new_x , new_y = mnist.train.next_batch(batch_size)
				_ , c = sess.run([optimizer,cost], feed_dict = {x: new_x, y: new_y})
				epoch_loss += c 
			print('Epochs', epochs, 'completed out of', display_epoch, 'cost:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('accuracy :', accuracy.eval({x : mnist.test.images, y: mnist.test.labels}))

training_model(x)
