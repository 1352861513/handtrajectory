import tensorflow as tf

BATCH_SIZE = 32


def _variable_with_weight_decay(shape, stddev, wd, name):
    # Genetate value weight and added to weight decay L2.
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev),
                      dtype=tf.float32,
                      name=name)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(data, prob):
    # Hidden Layer 1
    data_shape = data.get_shape().as_list()
    shape = [data_shape[1], 16]
    W_hidden1 = _variable_with_weight_decay(shape=shape,
                                        stddev=0.01,
                                        wd=5e-4,
                                        name='W_hidden1')
    b_hidden1 = tf.Variable(tf.constant(0.0, shape=[16]),
                        dtype=tf.float32,
                        name='b_hidden1')
    h_hidden1 = tf.nn.relu(tf.matmul(data, W_hidden1) + b_hidden1)
    # Dropout Layer 1
    h_hidden1_drop = tf.nn.dropout(h_hidden1, prob, name='drop1')

    # Hidden Layer 2
    shape = [16, 6]
    W_hidden2 = _variable_with_weight_decay(shape=shape,
                                            stddev=0.01,
                                            wd=5e-4,
                                            name='W_hidden2')
    b_hidden2 = tf.Variable(tf.constant(0.0, shape=[6]),
                            dtype=tf.float32,
                            name='b_hidden2')
    h_hidden2 = tf.add(tf.matmul(h_hidden1_drop, W_hidden2), b_hidden2, name='h_hidden3')

    argmax = tf.nn.softmax(h_hidden2, name='argmax')


    return h_hidden2

def total_loss(logits, labels):
    entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits))
    tf.add_to_collection('losses', entropy)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(loss, learning_rate, batch):
    # Train the net with stochastic gradient descent.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    return optimizer.minimize(loss, global_step=batch)


