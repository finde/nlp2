from __future__ import division
from operator import itemgetter
import collections
import helpers as utils
import decimal
from decimal import Decimal as D
import time
import cPickle
import numpy as np
from sklearn.metrics import mean_squared_error

# set decimal context
decimal.getcontext().prec = 4
decimal.getcontext().rounding = decimal.ROUND_HALF_UP


def _constant_factory(value):
    """
    define a local function for uniform probability initialization
    """
    return lambda: value


def _pprint(tbl):
    """
    print top 10 translation key-pairs
    :param tbl:
    :return:
    """
    for (e, f), v in sorted(tbl.items(), key=itemgetter(1), reverse=True)[:10]:
        print "p(%s|%s) = %f" % (e, f, v)


class IBMModel1:
    def __init__(self):
        self.t = collections.defaultdict()
        pass

    def get_argmax_t(self):
        return [_ for _ in sorted(self.t.items(), key=itemgetter(1), reverse=True)]

    def save_translation_table(self, filename='model1.p'):
        print self.get_argmax_t()
        cPickle.dump(self.get_argmax_t(), open(filename, 'w'))

    def load_translation_table(self, filename='model1.p'):
        return cPickle.load(self.get_argmax_t(), open(filename, 'r'))

    def _train(self, corpus, loop_count=1000, verbose=True, filename=None):
        """
         Pseudocode of EM for IBM Model 1 from Phillip Koehn's book

         initialize t(e|f) uniformly
         do until convergence
           set count(e|f) to 0 for all e,f
           set total(f) to 0 for all f
           for all sentence pairs (e_s,f_s)
             set total_s(e) = 0 for all e
             for all words e in e_s
               for all words f in f_s
                 total_s(e) += t(e|f)
             for all words e in e_s
               for all words f in f_s
                 count(e|f) += t(e|f) / total_s(e)
                 total(f)   += t(e|f) / total_s(e)
           for all f
             for all e
               t(e|f) = count(e|f) / total(f)

        :param corpus:
        :param loop_count:
        :param verbose:
        :return:
        """

        eps = 1E-7
        unit = len(corpus) / 10

        st = time.time()
        print "Reading keys ..."
        f_keys = set()
        for x, (es, fs) in enumerate(corpus):
            for f in fs:
                f_keys.add(f)

            if verbose and x % unit == 0:
                print '\t%d of %d' % (x, len(corpus))

        # todo:: make this faster by using counter ?

        # default value provided as uniform probability)
        self.t = collections.defaultdict(_constant_factory(D(1 / len(f_keys))))
        print "... done in: %.2fs" % (time.time() - st)

        # loop
        for i in range(loop_count):
            print '===== %s =====' % str(i + 1)

            if i > 0:
                _t = [_ for _ in self.t.values()]

            st = time.time()
            count = collections.defaultdict(D)
            total = collections.defaultdict(D)
            total_s = collections.defaultdict(D)
            for x, (es, fs) in enumerate(corpus):

                # added null
                es = [''] + es

                # compute normalization
                for e in es:
                    total_s[e] = D()
                    for f in fs:
                        total_s[e] += self.t[(e, f)]
                for e in es:
                    for f in fs:
                        count[(e, f)] += self.t[(e, f)] / total_s[e]
                        total[f] += self.t[(e, f)] / total_s[e]

                if verbose and (x % unit == 0):
                    print '\t%d of %d' % (x, len(corpus))

            # estimate probability
            for (e, f) in count.keys():
                self.t[(e, f)] = count[(e, f)] / total[f]

            if verbose:
                print "Iteration %s: %.2fs" % (str(i + 1), (time.time() - st))
                _pprint(self.t)

            if i > 0:
                __t = [_ for _ in self.t.values()]
                # TODO:: should based on perplexity
                converge = mean_squared_error(np.array(_t), np.array(__t))

                if converge < eps:
                    print '.. converged ..'
                    break

        self.save_translation_table(filename=filename)
        return self.t

    def train(self, source_file, target_file, max_lines=None, verbose=False):
        sentences = []

        s = time.time()
        with open(source) as file1, open(target) as file2:
            for source_sentences, target_sentences in zip(file1, file2):
                sentences.append([source_sentences, target_sentences])

                if max_lines and len(sentences) == max_lines:
                    break

        corpus = utils.mkcorpus(sentences)
        if verbose:
            print "Number of sentences: %d" % len(sentences)
            print "Time spent to read data: %.2fs" % (time.time() - s)

        self._train(corpus, loop_count=50, verbose=verbose)


if __name__ == '__main__':
    source = '../data/hansards.e'
    target = '../data/hansards.f'
    ibm = IBMModel1()
    ibm.train(source_file=source, target_file=target, max_lines=10)
    print ibm.get_argmax_t()