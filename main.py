#Example ‒ One-to-Many – learning to generate text
import os
import numpy as np
import re
import shutil
import tensorflow as tf
DATA_DIR = "./data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
def download_and_read(urls):
 texts = []
 for i, url in enumerate(urls):
     p = tf.keras.utils.get_file("ex1-{:d}.txt".format(i), url,
     cache_dir=".")
     text = open(p, "r", encoding="utf8").read()
     # remove byte order mark
     text = text.replace("\ufeff", "")
     # remove newlines
     text = text.replace('\n', ' ')
     text = re.sub(r'\s+', " ", text)
     # add it to the list
     texts.extend(text)
 return texts
texts = download_and_read([
 "http://www.gutenberg.org/cache/epub/28885/pg28885.txt",
 "https://www.gutenberg.org/files/12/12-0.txt"
])

# create the vocabulary
vocab = sorted(set(texts))
print("vocab size: {:d}".format(len(vocab)))
# create mapping from vocab chars to ints
char2idx = {c:i for i, c in enumerate(vocab)}
idx2char = {i:c for c, i in char2idx.items()}

# numericize the texts
texts_as_ints = np.array([char2idx[c] for c in texts])
data = tf.data.Dataset.from_tensor_slices(texts_as_ints)
# number of characters to show before asking for prediction
# sequences: [None, 100]
seq_length = 100
sequences = data.batch(seq_length + 1, drop_remainder=True)
def split_train_labels(sequence):
 input_seq = sequence[0:-1]
 output_seq = sequence[1:]
 return input_seq, output_seq
sequences = sequences.map(split_train_labels)
# set up for training
# batches: [None, 64, 100]
batch_size = 64
steps_per_epoch = len(texts) // seq_length // batch_size
dataset = sequences.shuffle(10000).batch(
 batch_size, drop_remainder=True)



