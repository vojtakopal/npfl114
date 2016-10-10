from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf

LABELS = 10
WIDTH = 28
HEIGHT = 28
HIDDEN = 100

class Network:
    def __init__(self, logdir, experiment, threads):
        def linear(x, layer_size, activation, name):
            with tf.variable_scope(name):
                input_size = x.get_shape()[-1]
                weights = tf.get_variable("weights", [input_size, layer_size])
                tf.histogram_summary("training/" + name + "/weights", weights)
                biases = tf.get_variable("biases", [layer_size])
                tf.histogram_summary("training/" + name + "/biases", biases)
                y = tf.matmul(x, weights) + biases
                if activation:
                    tf.histogram_summary("training/" + name + "/preactivation", y)
                    y = activation(y)
                tf.histogram_summary("training/" + name + "/value", y)
                return y

        # Construct the graph
        with tf.name_scope("inputs"):
            self.images = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            flattened_images = tf.reshape(self.images, [-1, WIDTH*HEIGHT], name="flattened_images")

        hidden_layer = linear(flattened_images, HIDDEN, activation=tf.tanh, name="hidden_layer")
        output_layer = linear(hidden_layer, LABELS, activation=None, name="output_layer")

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output_layer, self.labels), name="loss")
            tf.scalar_summary("training/loss", loss)
        with tf.name_scope("train"):
            self.training = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope("accuracy"):
            predictions = tf.argmax(output_layer, 1, name="predictions")
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, self.labels), tf.float32), name="accuracy")
            tf.scalar_summary("training/accuracy", accuracy)

        # Summaries
        self.summaries = {'training': tf.merge_all_summaries() }
        for dataset in ["dev", "test"]:
            self.summaries[dataset] = tf.scalar_summary(dataset + "/accuracy", accuracy)

        # Create the session
        self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                        intra_op_parallelism_threads=args.threads))

        self.session.run(tf.initialize_all_variables())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, experiment), graph=self.session.graph, flush_secs=10)
        self.steps = 0

    def train(self, images, labels):
        self.steps += 1
        feed_dict = {self.images: images, self.labels: labels}

        if self.steps == 1:
            metadata = tf.RunMetadata()
            self.session.run(self.training, feed_dict, options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata = metadata)
            self.summary_writer.add_run_metadata(metadata, 'step1')
        elif self.steps % 100 == 0:
            _, summary = self.session.run([self.training, self.summaries['training']], feed_dict)
            self.summary_writer.add_summary(summary, self.steps)
        else:
            self.session.run(self.training, feed_dict)

    def evaluate(self, dataset, images, labels):
        summary = self.summaries[dataset].eval({self.images: images, self.labels: labels}, self.session)
        self.summary_writer.add_summary(summary, self.steps)


if __name__ == '__main__':
    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=50, type=int, help='Batch size.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs.')
    parser.add_argument('--logdir', default="logs", type=str, help='Logdir name.')
    parser.add_argument('--exp', default="3-mnist-run-metadata-and-histograms", type=str, help='Experiment name.')
    parser.add_argument('--threads', default=1, type=int, help='Maximum number of threads to use.')
    args = parser.parse_args()

    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist_data/", reshape=False)

    # Construct the network
    network = Network(logdir=args.logdir, experiment=args.exp, threads=args.threads)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        network.evaluate("test", mnist.test.images, mnist.test.labels)
