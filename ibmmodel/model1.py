from __future__ import division
from operator import itemgetter
import collections
import helpers as utils
import decimal
from decimal import Decimal as D
import time
import cPickle

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
        pass

    @staticmethod
    def train(corpus, loop_count=1000, verbose=True):
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

        unit = len(corpus) / 100

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
        t = collections.defaultdict(_constant_factory(D(1 / len(f_keys))))
        print "... done in: %.2fs" % (time.time() - st)

        # loop
        for i in range(loop_count):
            print '===== %s =====' % str(i + 1)

            st = time.time()
            count = collections.defaultdict(D)
            total = collections.defaultdict(D)
            total_s = collections.defaultdict(D)
            for x, (es, fs) in enumerate(corpus):

                # added null
                # es = [''] + es

                # compute normalization
                for e in es:
                    total_s[e] = D()
                    for f in fs:
                        total_s[e] += t[(e, f)]
                for e in es:
                    for f in fs:
                        count[(e, f)] += t[(e, f)] / total_s[e]
                        total[f] += t[(e, f)] / total_s[e]

                if verbose and (x % unit == 0):
                    print '\t%d of %d' % (x, len(corpus))

            # estimate probability
            for (e, f) in count.keys():
                t[(e, f)] = count[(e, f)] / total[f]

            if verbose:
                print "Iteration %s: %.2fs" % (str(i + 1), (time.time() - st))
                _pprint(t)

        return t


if __name__ == '__main__':

    source = '../data/hansards.e'
    target = '../data/hansards.f'
    ibm = IBMModel1()

    sentences = []
    MAX_LINES = 1000

    s = time.time()
    with open(source) as file1, open(target) as file2:
        for source_sentences, target_sentences in zip(file1, file2):
            sentences.append([source_sentences, target_sentences])

            if MAX_LINES and len(sentences) == MAX_LINES:
                break

    corpus = utils.mkcorpus(sentences)
    print "Number of sentences: %d" % len(sentences)
    print "Time spent to read data: %.2fs" % (time.time() - s)

    t = ibm.train(corpus, loop_count=50, verbose=True)
    _pprint(t)

    cPickle.dump(t, open('trained_t_', 'w'))