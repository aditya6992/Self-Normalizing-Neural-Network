from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler


# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 30
batch_size = 64
display_step = 1

# Network Parameters
n_hidden_1 = 784 # 1st layer number of features
n_hidden_2 = 784 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropoutRate = tf.placeholder(tf.float32)

# (1) Scale input to zero mean and unit variance
scaler = StandardScaler().fit(mnist.train.images)

# Tensorboard
logs_path = '~/tmp'


# Create model
def multilayer_perceptron(x, weights, biases, rate):

    rate = 1.0 - rate

    # layer_1 with SELU activation with alpha dropout as mentioned in the paper
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.selu(layer_1)
    layer_1 = tf.contrib.nn.alpha_dropout(layer_1, rate)

    # Hidden layer 2 with SELU activation with alpha dropout as mentioned in the paper
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.selu(layer_2)
    layer_2 = tf.contrib.nn.alpha_dropout(layer_2, rate)


    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=np.sqrt(1 / n_input))),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=np.sqrt(1 / n_hidden_1))),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=np.sqrt(1 / n_hidden_2)))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, rate=dropoutRate)

# Define cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a histogramm for weights
tf.summary.histogram("weights2", weights['h2'])
tf.summary.histogram("weights1", weights['h1'])

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)

# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = scaler.transform(batch_x)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y, dropoutRate: 0.5})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            accTrain, costTrain, summary = sess.run([accuracy, cost, merged_summary_op],
                                                    feed_dict={x: batch_x, y: batch_y,
                                                               dropoutRate: 0.0})
            summary_writer.add_summary(summary, epoch)

            print("Train-Accuracy:", accTrain, "Train-Loss:", costTrain)

            batch_x_test, batch_y_test = mnist.test.next_batch(512)
            batch_x_test = scaler.transform(batch_x_test)

            accTest, costVal = sess.run([accuracy, cost], feed_dict={x: batch_x_test, y: batch_y_test,
                                                                     dropoutRate: 0.0})

            print("Validation-Accuracy:", accTest, "Val-Loss:", costVal, "\n")
