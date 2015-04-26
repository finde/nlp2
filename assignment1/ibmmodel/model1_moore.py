from __future__ import division
from ibmmodel import *


class IBMModel1Moore(IBMModel):
    """
        Moore (2004) proposes to improve IBM model 1 by:
        1. smoothing counts for rare words
        2. assigning more probability mass to Null alignments
        3. heuristic initialisation of lexical parameters
    """

    def __init__(self, source_corpus, target_corpus, verbose=False, init='moore',
                 smooth_n=1, smooth_v=100000, n_null=3):
        self.source_corpus = source_corpus
        self.target_corpus = target_corpus

        self.init = init
        self.verbose = verbose

        # small number 0 --> 0.005
        self.smooth_n = smooth_n
        self.smooth_v = smooth_v
        self.n_null = n_null

        # init t prob
        # t(f|e) = t[e][f]
        self.t = defaultdict(dd)
        e_word = list(set(reduce(operator.add, target_corpus)))
        if init.startswith('uniform'):
            print "Init T-table: uniform"
            for f_sent, e_sent in zip(self.source_corpus, self.target_corpus):
                for f_i in f_sent:
                    for e_j in [None] + e_sent:
                        self.t[e_j][f_i] = 1.0 / len(e_word)

        elif init.startswith('random'):
            print "Init T-table: random"
            for f_sent, e_sent in zip(self.source_corpus, self.target_corpus):
                for f_i in f_sent:
                    for e_j in [None] + e_sent:
                        self.t[e_j][f_i] = random.random() * -1 + 1

        else:
            print "Init T-table using: preset data"
            self.t = init

    def get_sent_loglikelihood(self, e_sent, f_sent):
        l_f = len(f_sent)
        l_e = len(e_sent)
        ll = -(l_e * np.log(l_f + 1))  # normalizing constant

        for f_i in f_sent:
            t_ef = 0
            for e_j in ([None] * self.n_null + e_sent):
                t_ef += self.t[e_j][f_i]
            ll += np.log(t_ef)

        return ll

    def get_sent_alignment(self, e_sent, f_sent, k):
        align = []
        for i, f_i in enumerate(f_sent):
            probs = [self.t[e_j][f_i] for j, e_j in enumerate([None] + e_sent)]
            j = np.argmax(np.array(probs))

            if j != 0:
                align.append(" ".join(['%04d' % (k + 1), str(i + 1), str(j)]))

        return align

    def train(self, max_iter=np.inf, eps=1E-1, log_file='ibm_model_1_moore', test_set=None):
        it = 0
        delta_ll = np.inf
        ll = -np.inf

        # remove old log
        if log_file:
            clean_log(log_file + "_perp")
            clean_log(log_file + "_ll")

        _init_time = time.time()
        # loop until converged or until reach maximum iteration
        while it < max_iter and delta_ll > eps:
            it += 1
            start = time.time()
            print "---Iteration: %s---" % it

            # set all counts to zero
            # + NULL
            c_ef = Counter()
            c_e = Counter()
            for f_sent, e_sent in zip(self.source_corpus, self.target_corpus):
                for f_i in f_sent:
                    sum_t = sum([self.t[e_j][f_i] for e_j in [None] * self.n_null + e_sent]) * 1.0
                    for e_j in [None] + e_sent:
                        delta = self.t[e_j][f_i] / sum_t
                        c_ef[(e_j, f_i)] += delta
                        c_e[e_j] += delta

            # update t
            # + smoothing
            for (e, f), c in c_ef.items():
                self.t[e][f] = (c + self.smooth_n) / (c_e[e] + self.smooth_n * self.smooth_v)

            old_ll = ll
            ll = self.get_loglikelihood()
            perp = self.get_perplexity()

            # save into log
            if log_file:
                write_to_log(log_file + "_ll", ll)
                write_to_log(log_file + "_perp", perp)

            print "   LogLikelihood: %s" % ll
            print "   Perplexity: %s" % perp
            print "   Iteration time: %s" % str(time.time() - start)
            print "   Elapsed time: %s" % str(time.time() - _init_time)
            print ""
            delta_ll = ll - old_ll

            if it % 10 == 0:
                self.dump('cache/%s_ef_%s.%s' % (log_file, self.init, str(it)))
                if test_set is not None:
                    self.get_alignments(sentences_pair=test_set,
                                        log_file='results/%s_ef_%s_align.%s' % (log_file, self.init, str(it)))

                # plot_likelihood('Log-Likelihood IBM Model 1 (%s)' % init,
                #                 'results/' + log_file + "_ll.txt",
                #                 '%s_ef_%s' % (log_file, init))

        return self.t


if __name__ == '__main__':

    if len(sys.argv) > 1:
        train_source = sys.argv[1]
    else:
        train_source = 'data/hansards.36.2.e'

    if len(sys.argv) > 2:
        train_target = sys.argv[2]
    else:
        train_target = 'data/hansards.36.2.f'

    if len(sys.argv) > 3:
        test_source = sys.argv[3]
    else:
        test_source = 'data/test.e'

    if len(sys.argv) > 4:
        test_target = sys.argv[4]
    else:
        test_target = 'data/test.f'

    if len(sys.argv) > 5:
        init = sys.argv[5]
    else:
        init = 'uniform'

    if len(sys.argv) > 6:
        max_lines = int(sys.argv[6])
    else:
        max_lines = np.inf

    if len(sys.argv) > 7:
        smooth_n = float(sys.argv[7])
    else:
        smooth_n = 0.01

    if len(sys.argv) > 8:
        smooth_v = int(sys.argv[8])
    else:
        smooth_V = 0

    if len(sys.argv) > 9:
        n_null = int(sys.argv[9])
    else:
        n_null = 3


    # todo remove the log file / remove the cache file
    s, t = get_sentences_pair(train_source, train_target, max_lines=max_lines)
    _s, _t = get_sentences_pair(test_source, test_target)

    print "init..."

    # baseline
    model = IBMModel1Moore(source_corpus=s + _s, target_corpus=t + _t, init=init,
                           smooth_n=smooth_n, smooth_v=smooth_v, n_null=n_null)

    print "training..."
    model.train(test_set=zip(_s, _t), log_file='ibm_model_1_moore_%s' % init, eps=1)

    model.dump('cache/ibm_model_1_moore_ef_%s' % init)

    model.get_alignments(sentences_pair=zip(_s, _t), log_file='alignments.out')

    # plot_likelihood('results/ibm_model_1_moore_%s_ll.txt' % init,
    # 'ibm_model_1_moore_ef_%s' % init)

    # print T