# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec mini-batched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does traditional minibatching.

The key ops used are:
* placeholder for feeding in tensors for each example.
* embedding_lookup for fetching rows from the embedding matrix.
* sigmoid_cross_entropy_with_logits to calculate the loss.
* GradientDescentOptimizer for optimizing the loss.
* skipgram custom op that does input processing.
"""

import torch
import os
import tensorflow as tf
import warnings

word2vec = tf.load_op_library(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'word2vec_ops.so'))
warnings.filterwarnings("ignore")


class Word2Vec_Skipgram_Data(object):
    """Word2Vec model (Skipgram)."""

    def __init__(
            self,
            train_data,
            num_samples,
            batch_size,
            window_size,
            min_count,
            subsample,
            session):
        self.train_data = train_data
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.window_size = window_size
        self.min_count = min_count
        self.subsample = subsample
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()

    def build_graph(self):
        """Build the graph for the full model."""
        # The training data. A text file.
        (words, counts, words_per_epoch, self._epoch, self._words, examples,
         labels) = word2vec.skipgram_word2vec(filename=self.train_data,
                                              batch_size=self.batch_size,
                                              window_size=self.window_size,
                                              min_count=self.min_count,
                                              subsample=self.subsample)
        (self.vocab_words, self.vocab_counts,
         self.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        self.vocab_size = len(self.vocab_words)
        print("Data file: ", self.train_data)
        print("Vocab size: ", self.vocab_size - 1, " + UNK")
        print("Words per epoch: ", self.words_per_epoch)
        self._examples = examples
        self._labels = labels
        self._id2word = self.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

        id2word = []
        for i, w in enumerate(self._id2word):
            try:
                id2word.append(int(w))
            except BaseException:
                id2word.append(w)

        self._id2word = id2word

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [self.batch_size, 1])
        # Negative sampling.
        self.sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.num_samples,
            unique=True,
            range_max=self.vocab_size,
            distortion=0.75,
            unigrams=self.vocab_counts.tolist()))

    def next_batch(self):
        """Train the model."""

        initial_epoch, e, l, s, words = self._session.run(
            [self._epoch, self._examples, self._labels, self.sampled_ids, self._words])

        # All + 1 because of the padding_idx
        e_new = []
        for e1 in e:
            e_new.append(self._id2word[e1] + 1)

        label_new = []
        for l1 in l:
            label_new.append(self._id2word[l1] + 1)

        sampled_id_new = []
        for s1 in s:
            sampled_id_new.append(self._id2word[s1] + 1)

        return e_new, label_new, sampled_id_new, initial_epoch, words



