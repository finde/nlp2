from __future__ import division
from operator import itemgetter
import collections
import cPickle
import operator
import numpy as np


def _pprint(tbl):
    """
    print top 10 translation key-pairs
    :param tbl:
    :return:
    """
    for (e, f), v in sorted(tbl.items(), key=itemgetter(1), reverse=True)[:10]:
        print "p(%s|%s) = %f" % (e, f, v)


class IBMModel1:
    def __init__(self, source_corpus, target_corpus):
        self.t = collections.defaultdict()
        self.q = collections.defaultdict()

        self.source_corpus = source_corpus
        self.target_corpus = target_corpus
        self.f_word = list(set(reduce(operator.add, source_corpus)))
        self.e_word = list(set(reduce(operator.add, target_corpus)))
        pass

    def save_translation_table(self, filename='model1.p'):
        print self.get_argmax_t()
        cPickle.dump(self.get_argmax_t(), open(filename, 'w'))

    def load_translation_table(self, filename='model1.p'):
        return cPickle.load(self.get_argmax_t(), open(filename, 'r'))

    def get_viterbi(self, source_sentence, target_sentence):

        pass

    def train(self, max_iter=5):
        f_word = self.f_word
        e_word = self.e_word
        t = self.t
        # q = self.q
        old_ll = 0
        eps = 1E-2

        for k in range(max_iter):
            C = {}
            for m, l in zip(self.source_corpus, self.target_corpus):
                l = ['NULL'] + l
                if k == 0:
                    for fi in m:
                        for ej in l:
                            if "%s|%s" % (fi, ej) not in t:
                                t["%s|%s" % (fi, ej)] = 1.0 / len(e_word)

                for i, fi in enumerate(m):
                    sum_t = sum([t["%s|%s" % (fi, ej)] for ej in l]) * 1.0
                    for j, ej in enumerate(l):
                        delta = t["%s|%s" % (fi, ej)] / sum_t

                        C["%s %s" % (ej, fi)] = C.get("%s %s" % (ej, fi), 0) + delta
                        C["%s" % (ej)] = C.get("%s" % (ej), 0) + delta

                        # C["%s|%s %s %s" % (j, i, len(m), len(l))] = C.get("%s|%s %s %s" % (j, i, len(m), len(l)), 0) + delta
                        # C["%s %s %s" % (j, len(m), len(l))] = C.get("%s %s %s" % (j, len(m), len(l)), 0) + delta

            # set t
            for f in f_word:
                for e in e_word:
                    if "%s %s" % (e, f) in C and "%s" % (e) in C:
                        t["%s|%s" % (f, e)] = C["%s %s" % (e, f)] / C["%s" % (e)]

            ll = 0
            for m, l in zip(self.source_corpus, self.target_corpus):
                l = ['NULL'] + l
                for i, fi in enumerate(m):
                    _t_ef = 0

                    for j, ej in enumerate(l):
                        # num = C.get("%s|%s %s %s" % (j, i, len(m), len(l)), 0)
                        # denum = C.get("%s %s %s" % (j, len(m), len(l)), 0)
                        # q["%s|%s %s %s" % (j, i, len(m), len(l))] = num / denum

                        # likelihood
                        _t_ef += t["%s|%s" % (fi, ej)]

                    ll += np.log(_t_ef)

                ll -= len(l) * np.log(len(m))

            print "---em iteration:%s---: %s" % (k + 1, ll)

            with open('ll_ibm_model_1.txt', 'ab') as f:
                f.write(str(ll) + '\n')
                f.close()

            if abs(old_ll - ll) < eps:
                break

            old_ll = ll

        self.t = t
        # self.q = q

        with open('t_ibm_model_1', 'w') as f:
            cPickle.dump({'t': self.t, 'it': k, 'll': ll}, f)
            f.close()


if __name__ == '__main__':
    source = '../data/hansards.36.2.e'
    target = '../data/hansards.36.2.f'
    # source = '../data/corpus_1000.nl'
    # target = '../data/corpus_1000.en'

    source_sentences = []
    target_sentences = []
    max_lines = None
    with open(source) as file_f, open(target) as file_e:
        for _f, _e in zip(file_f, file_e):
            source_sentences.append(_f.rstrip('\n').split(' '))
            target_sentences.append(_e.rstrip('\n').split(' '))

            if len(source_sentences) % 1000 == 0:
                print "Read: %s lines" % (len(source_sentences))

            if max_lines:
                if len(source_sentences) == max_lines:
                    break

    with open('ll_ibm_model_1.txt', 'w') as f:
        f.close()

    ibm = IBMModel1(source_corpus=source_sentences, target_corpus=target_sentences)
    ibm.train(max_iter=10000)
    # T = ibm.get_viterbi()
    # print T