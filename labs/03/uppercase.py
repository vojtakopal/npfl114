#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

def to_categorical(labels):
    return tf.keras.utils.to_categorical(y=labels, num_classes=2)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=None, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=None, type=int, help="Window size to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

# Load data
uppercase_data = UppercaseData(args.window, args.alphabet_size)
print('alphabet', uppercase_data.test.alphabet)

regulizer = None
if args.l2 != 0:
    regulizer = tf.keras.regularizers.L1L2(l2=args.l2)

dropout = tf.keras.layers.Dropout(rate=args.dropout)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
    tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    tf.keras.layers.Flatten(),
    dropout,
])

for hidden_layer in args.hidden_layers:
    model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu, kernel_regularizer=regulizer, bias_regularizer=regulizer))
    model.add(dropout)

model.add(tf.keras.layers.Dense(2, kernel_regularizer=regulizer, bias_regularizer=regulizer))

print('Model set up')

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
)

print('Model compiled')

train_subset_size=200000

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None
model.fit(
    uppercase_data.train.data["windows"][:train_subset_size], to_categorical(uppercase_data.train.data["labels"][:train_subset_size]),
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(uppercase_data.dev.data["windows"], to_categorical(uppercase_data.dev.data["labels"])),
    callbacks=[tb_callback],
)

print('Model fit')

# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.

with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.
    alphabet = uppercase_data.test.alphabet
    text_size = len(uppercase_data.test.text)
    print('building text')
    last = 0
    for i in range(0, text_size):
        li = i - 1*args.window
        ri = i + 1*args.window + 1
        word = uppercase_data.test.text[max(0, li):ri].rjust(1 + 2 * args.window)
        vec = np.array([[alphabet.index(c) if c in alphabet else 0 for c in word]])
        val = model.predict(vec)
        val_index = np.argmax(val)
        char = uppercase_data.test.text[i]

        if val_index == 1:
            char = char.upper()

        out_file.write(char)

        progress = round(100 * i / text_size)
        if progress > last:
            last = progress
            print('.')
