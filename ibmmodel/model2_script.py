from ibmmodel import *
from model1 import IBMModel1
from model2 import IBMModel2

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

    s, t = get_sentences_pair(train_source, train_target, max_lines=max_lines)
    _s, _t = get_sentences_pair(test_source, test_target)

    print "init... IBM model 1 = 10 iter"
    model = IBMModel1(source_corpus=s + _s, target_corpus=t + _t, init=init)
    t_prob = model.train(test_set=zip(_s, _t), log_file='ibm_model_1_%s' % init, max_iter=10, eps=1)
    model.get_alignments(sentences_pair=zip(_s, _t), log_file='results/ibm_model_2_ef_%s_1_align' % init)

    model2 = IBMModel2(source_corpus=s + _s, target_corpus=t + _t, init='ibm1', init_t=t_prob)
    model2.train(test_set=zip(_s, _t), log_file='ibm_model_2_%s' % 'ibm1', eps=1)
    model2.get_alignments(sentences_pair=zip(_s, _t), log_file='results/ibm_model_2_ef_%s_2_align' % init)

    # plot_likelihood('results/ibm_model_2_%s_ll.txt' % init,
    # 'ibm_model_2_ef_%s' % init)
    # print T