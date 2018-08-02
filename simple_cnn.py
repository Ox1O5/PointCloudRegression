import tensorflow as tf
import numpy as np
import utils
import os


def check_accuracy(sess, dset, x, scores, is_training=None):
    val_batches, val_l = dset
    feed_dict = {x: val_batches, is_training: 0}
    scores_np = sess.run(scores, feed_dict=feed_dict)
    dist = sess.run(tf.reduce_mean(tf.abs(val_l-scores_np))*180)
    print(scores_np)
    print(val_l)
    print('batch_dist = %.4f' % dist)


def model_init_fn(inputs, is_training):
    channel_1, channel_2, channel_3, channel_4, channel_5, outputs = 128, 128, 64, 64, 32, 2
    pool_size = 2
    input_shape = (224, 224, 3)
    initializer = tf.variance_scaling_initializer(1.0)
    layers = [
        tf.layers.Conv2D(input_shape=input_shape,
                         filters=channel_1, kernel_size=3, padding='valid',
                         use_bias=True, bias_initializer=initializer,
                         activation=tf.nn.relu,
                         kernel_initializer=initializer),
        tf.layers.MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        tf.layers.Conv2D(filters=channel_2, kernel_size=3, padding='valid',
                         use_bias=True, bias_initializer=initializer,
                         activation=tf.nn.relu,
                         kernel_initializer=initializer),
        tf.layers.MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        # tf.layers.Conv2D(filters=channel_3, kernel_size=3, padding='same',
        #                  use_bias=True, bias_initializer=initializer,
        #                  activation=tf.nn.relu,
        #                  kernel_initializer=initializer),
        # tf.layers.MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        tf.layers.Flatten(),
        tf.layers.Dense(channel_4, use_bias=True, bias_initializer=initializer,
                        activation=tf.nn.relu,
                        kernel_initializer=initializer),
        # tf.layers.Dense(channel_5, use_bias=True, bias_initializer=initializer,
        #                 activation=tf.nn.relu,
        #                 kernel_initializer=initializer),
        tf.layers.Dense(outputs, use_bias=True, bias_initializer=initializer,
                        kernel_initializer=initializer)
    ]
    model = tf.keras.Sequential(layers)
    return model(inputs)


def optimizer_init_fn():
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    optimizer = tf.train.AdamOptimizer(0.001)
    return optimizer


def train_part(model_init_fn, optimizer_init_fn, num_epochs=1):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, 2])
    is_training = tf.placeholder(tf.bool, name='is_training')
    scores = model_init_fn(x, is_training)
    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(labels=y/3.14*180, predictions=scores/3.14*180)
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        optimizer = optimizer_init_fn()
        train_op = optimizer.minimize(loss)

    img, label = utils.read_and_decode('train')
    img_batches, label_batches = tf.train.batch([img, label], batch_size=20)#, capacity=2000,
                                                                         #min_after_dequeue=1000)
    val, l = utils.read_and_decode('valid')
    val_batches, val_label_batches = tf.train.batch([val, l], batch_size=20)#, capacity=1000,
                                                         #min_after_dequeue=500)
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./train', graph=tf.get_default_graph())
        threads = tf.train.start_queue_runners(sess=sess)
        val_value, val_label = sess.run([val_batches, val_label_batches])
        val_dset = (val_value, val_label)
        for epoch in range(num_epochs):
            x_np, y_np = sess.run([img_batches, label_batches])
            feed_dict = {x: x_np, y: y_np, is_training: 1}
            loss_np, summery, _ = sess.run([loss, merged, train_op], feed_dict=feed_dict)
            writer.add_summary(summery, epoch)
            if epoch % print_every == 0:
                print('Iteration %d, loss = %.4f' % (epoch, loss_np))
                check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                print()


if not os.path.exists('valid.tfrecords'):
    utils.img2tfrecord("train")
    utils.img2tfrecord("valid")

print_every = 400
num_epochs = 20000
train_part(model_init_fn, optimizer_init_fn, num_epochs)