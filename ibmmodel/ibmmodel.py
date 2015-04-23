from __future__ import division
from collections import Counter, defaultdict
import sys
import os
import cPickle
import random
import operator
import numpy as np
from matplotlib import pyplot as plt
import time


def dd():
    return defaultdict(int)


def get_sentences_pair(source_file, target_file, max_lines=None):
    source_sentences = []
    target_sentences = []

    with open(source_file) as file_f, open(target_file) as file_e:
        for _f, _e in zip(file_f, file_e):
            source_sentences.append(_f.rstrip('\n').split(' '))
            target_sentences.append(_e.rstrip('\n').split(' '))

            if len(source_sentences) % 1000 == 0:
                print "Read: %s lines" % (len(source_sentences))

            if max_lines:
                if len(source_sentences) == max_lines:
                    break

    return source_sentences, target_sentences


def plot_likelihood(title, file_name, save_to):
    with open(file_name, "r") as f:
        data = [float(x.strip()) for x in f.readlines()]

    plt.title(title)
    plt.plot(data)
    plt.xlabel("iterations")
    plt.ylabel("data log-likelihood")
    plt.savefig("results/" + save_to + "_plot.pdf")

    f.close()


def clean_log(log_file):
    if not os.path.exists('results'):
        os.makedirs('results')

    with open('results/' + log_file + '.txt', 'w') as f:
        f.close()


def write_to_log(log_file, text):
    if not os.path.exists('results'):
        os.makedirs('results')

    with open('results/' + log_file + '.txt', 'ab') as f:
        f.write(str(text) + '\n')
        f.close()


class IBMModel(object):
    """
    Abstract class for IBM Models.
    """

    @staticmethod
    def get_sent_loglikelihood(e_sent, f_sent):
        pass

    def get_loglikelihood(self):
        """
        Sum over log likelihood of each sentence pair
        """
        ll = 0
        for f_sent, e_sent in zip(self.source_corpus, self.target_corpus):
            ll += self.get_sent_loglikelihood(e_sent, f_sent)
        return ll

    def get_perplexity(self):
        ll = self.get_loglikelihood()
        return -ll

    def get_alignments(self, sentences_pair=None, log_file='align'):
        """
        Viterbi alignment
        """
        if sentences_pair is None:
            sentences_pair = zip(self.source_corpus, self.target_corpus)

        with open(log_file, 'w') as f:
            for k, (f_sent, e_sent) in enumerate(sentences_pair):
                sure, proba = self.get_sent_alignment(e_sent, f_sent, k)

                for s in sure:
                    f.write(s + '\n')

                for p in proba:
                    f.write(p + '\n')

        f.close()

    def dump(self, save_as_file):
        """
        Save the model into cache file
        """
        with open(save_as_file, 'w') as f:
            cPickle.dump(self.t, f)
            f.close()

    def load(self, load_from_file):
        with open(load_from_file, 'r') as f:
            self.t = cPickle.load(f)
            f.close()
            return self.t