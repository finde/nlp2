from ibmmodel import *


# todo load alignment from ibm 1
class IBMModel2(IBMModel):
    def __init__(self, source_corpus, target_corpus, verbose=False, init='uniform', init_t=''):
        self.source_corpus = source_corpus
        self.target_corpus = target_corpus

        self.verbose = verbose
        e_word = list(set(reduce(operator.add, target_corpus)))

        self.init = init

        # init t prob
        # t(f|e) = t[e][f]
        self.t = defaultdict(dd)
        self.q = defaultdict(dd)
        if init == 'uniform':
            print "Init T-table: uniform"
            for f_sent, e_sent in zip(self.source_corpus, self.target_corpus):
                l = len(e_sent)
                m = len(f_sent)
                for i, f_i in enumerate(f_sent):
                    for j, e_j in enumerate([None] + e_sent):
                        self.t[e_j][f_i] = 1.0 / len(e_word)
                        self.q[(i, l, m)][j] = 1.0 / len(e_word)

        elif init.startswith('random'):
            print "Init T-table: random"
            for f_sent, e_sent in zip(self.source_corpus, self.target_corpus):
                l = len(e_sent)
                m = len(f_sent)

                for i, f_i in enumerate(f_sent):
                    for j, e_j in enumerate([None] + e_sent):
                        self.t[e_j][f_i] = random.random() * -1 + 1
                        self.q[(i, l, m)][j] = random.random() * -1 + 1

        else:
            print "Init T-table using: preset data (ibm model 1)"
            self.t = init_t

            print "Init Q-table: uniform"
            for f_sent, e_sent in zip(self.source_corpus, self.target_corpus):
                l = len(e_sent)
                m = len(f_sent)
                for i, f_i in enumerate(f_sent):
                    for j, e_j in enumerate([None] + e_sent):
                        self.q[(i, l, m)][j] = 1.0 / len(e_word)


    def get_sent_loglikelihood(self, e_sent, f_sent):
        l = len(e_sent)
        m = len(f_sent)
        ll = -(l * np.log(m + 1))  # normalizing constant

        for i, f_i in enumerate(f_sent):
            t_ef = 0
            for j, e_j in enumerate([None] + e_sent):
                t_ef += self.t[e_j][f_i] * self.q[(i, l, m)][j]
            ll += np.log(t_ef)

        return ll

    def get_sent_alignment(self, e_sent, f_sent, k):
        l = len(e_sent)
        m = len(f_sent)
        align = []
        for i, f_i in enumerate(f_sent):
            probs = [self.t[e_j][f_i] * self.q[(i, l, m)][j] for j, e_j in enumerate([None] + e_sent)]
            j = np.argmax(np.array(probs))

            # print probs
            # print k, i, j
            if j != 0:
                align.append(" ".join(['%04d' % (k + 1), str(i + 1), str(j)]))

        return align

    def train(self, max_iter=np.inf, eps=1E-2, log_file='ibm_model_2', test_set=None):
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
            c_ef = Counter()
            c_e = Counter()
            c_jilm = Counter()
            c_ilm = Counter()
            for f_sent, e_sent in zip(self.source_corpus, self.target_corpus):
                l = len(e_sent)
                m = len(f_sent)
                for i, f_i in enumerate(f_sent):
                    sum_t = sum([self.t[e_j][f_i] for e_j in [None] + e_sent]) * 1.0
                    for j, e_j in enumerate([None] + e_sent):
                        delta = self.t[e_j][f_i] / sum_t
                        c_ef[(e_j, f_i)] += delta
                        c_e[e_j] += delta
                        c_jilm[(j, i, l, m)] += delta
                        c_ilm[(i, l, m)] += delta

            # update t
            for (e, f), c in c_ef.items():
                self.t[e][f] = c / c_e[e]

            # update q
            for (j, i, l, m), c in c_jilm.items():
                self.q[(i, l, m)][j] = c / c_ilm[(i, l, m)]

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
                self.dump('cache/ibm_model_2_ef_%s.%s' % (self.init, str(it)))
                if test_set is not None:
                    self.get_alignments(sentences_pair=test_set,
                                        log_file='results/ibm_model_2_ef_%s_align.%s' % (self.init, str(it)))

                    # plot_likelihood('Log-Likelihood IBM Model 2 (%s)' % init,
                    # 'results/' + log_file + "_ll.txt",
                    #                 'ibm_model_2_ef_%s' % init)

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
        init = 'random'

    if len(sys.argv) > 6:
        max_lines = int(sys.argv[6])
    else:
        max_lines = np.inf

    s, t = get_sentences_pair(train_source, train_target, max_lines=max_lines)
    _s, _t = get_sentences_pair(test_source, test_target)

    print "init..."
    model = IBMModel2(source_corpus=s + _s, target_corpus=t + _t, init=init)

    print "training..."
    model.train(test_set=zip(_s, _t), log_file='ibm_model_2_%s' % init, eps=1)

    model.dump('cache/ibm_model_2_ef_%s' % init)

    model.get_alignments(sentences_pair=zip(_s, _t), log_file='alignments.out')

    # plot_likelihood('results/ibm_model_2_%s_ll.txt' % init,
    # 'ibm_model_2_ef_%s' % init)
    # print T