__author__ = 'finde'

from model1 import IBMModel1

class IBMModel2:

    def __init__(self):
        pass

    def train(self, sentences_pair):
        """
        input: set of sentence pairs (e,f)
        output: probability distributions of t (lexical translation) and a (alignment)
        pseudocode:
            carry over t(e|f) from Model 1
            initialize a(i|j,le,lf) = 1/(lf+1) for all i,j,le,lf
            while not converged do:
                // initialize
                count(e|f)=0 for all e,f
                total(f)=0 for all f
                counta(i|j,le,lf)=0 for all i,j,le,lf
                totala(j,le,lf)=0 for all j,le,lf

                for all sentence pairs (e,f) do
                    le = length(e), lf = length(f)

                    // compute normalization 12:
                    for j =1.. le do // all word positions in e
                        s-total(ej)=0

                        for i =0.. lf do // all word positions in f
                            s-total(ej)+=t(ej|fi) ∗ a(i|j,le,lf)
                        end for

                    end for

                    // collect counts
                    for j =1.. le do // all word positions in e

                        for i =0.. lf do // all word positions in f
                            c = t(ej|fi) ∗ a(i|j,le,lf) / s-total(ej)
                            count(ej|fi)+= c
                            total(fi)+= c
                            counta(i|j,le,lf)+= c
                            totala(j,le,lf)+= c
                        end for

                    end for

                end for

                // estimate probabilities
                t(e|f)=0 for all e,f
                a(i|j,le,lf)=0 for all i,j,le,lf

                for all e,f do
                    t(e|f) = count(e|f) / total(f)
                end for

                for all i,j,le,lf do
                    a(i|j,le,lf) = counta(i|j,le,lf) / totala(j,le,lf)
                end for
            end while
        """

