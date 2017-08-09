'''
input + weight 
> activation function
> hidden layer 1 
> hidden layer 1 + weights 
> activation function
> hidden layer 2 
> hidden layer 2 + weights
> output layer

compare output to intended output 
> cost function(cross entropy)
> optimization function (optimizer)
> minimize cost (adamPptimizer ... SGD, AdaGrad)

feed forward + back propagation = epoch
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
# number of node in each layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# number of classes
n_class = 10
# each time read 100 data
batch_size = 100

# set data shape(optional), if set the when data
# not satisfied it will throw an error

# here "None" is the number of instance, it will automate calculate the number
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    # set weights and bias for each layer
    hidden_layer_1 = {
        # digit picture are 28 *28 pix, hence the input data are 784 feature(row)
        # the number of nodes(column) is n_nodes_hl1
        'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
        # each nodes in hidden layer need their own bias variable, notice here the
        # Variable should be a matrix
        'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer_2 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_layer_3 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_class])),
        'bias': tf.Variable(tf.random_normal([n_class]))}
    
    # activeFunc(data * weights + bias) = hidden_layer_1
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), 
                hidden_layer_1['bias'])
    l1 = tf.nn.relu(l1) # activation function
    
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), 
                hidden_layer_2['bias'])
    l2 = tf.nn.relu(l2) # activation function
    
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), 
                hidden_layer_3['bias'])
    l3 = tf.nn.relu(l3) # activation function
    
    # calculate the output layer
    output = tf.add(tf.matmul(l3, output_layer['weights']),
                    output_layer['bias'])
    
    return output

def train_neural_network(x):
    # call your customized neural_network_model
    prediction = neural_network_model(x)
    # set the cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, 
                                                                  labels = y) )
    # set the optimizer for neural network
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # number of cycles of (feed forward + back propagation)
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        # loop for neural network train
        for epoch in range(hm_epochs):
            # in each generation initialize the loss to be 0
            epoch_loss = 0
            # SGD used here each time use "batch_size" instance to calculate the 
            # gradient descend
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # get the data and the label from the data set
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # calculate the cost, use the defined optimizer and cost function
                # we need to set the x and y specific
                _, c = sess.run([optimizer, cost], 
                                feed_dict = {x: epoch_x, y: epoch_y})
                # sum up the loss for the whole generation
                epoch_loss += c
            # print out the loss for each generation
            print('Epoch', (epoch + 1), ' completed out of', hm_epochs,
                  ' loss:', epoch_loss)
        
        # calculate the correctness and print it out
        correct = tf.equal(tf.argmax(prediction, 1), 
                           tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy: ", accuracy.eval(
            {x:mnist.test.images,
             y:mnist.test.labels}))

# run the whole neural network
train_neural_network(x)



