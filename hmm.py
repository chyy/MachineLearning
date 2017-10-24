# coding:utf-8


import numpy as np


class HMM(object):
    def __init__(self, init_prob, trans_prob, emit_prob):
        self.initial_prob = init_prob
        self.transition_prob = trans_prob
        self.emission_prob = emit_prob
        self.hidden_var = init_prob.keys()
        self.hidden_seq = []
        self.observation_var = emit_prob.keys()
        self.observation_seq = []

    def sample_sequence(self, length):
        h_var = sampling(self.initial_prob)
        self.hidden_seq.append(h_var)

        e_prob = self.emission_prob[h_var]
        o_var = sampling(e_prob)
        self.observation_seq.append(o_var)

        for i in xrange(1, length):
            t_prob = self.transition_prob[h_var]
            e_prob = self.emission_prob[h_var]

            h_var = sampling(t_prob)
            self.hidden_seq.append(h_var)

            o_var = sampling(e_prob)
            self.observation_seq.append(o_var)

    def decoding(self, seq):
        seq_len = len(seq)
        hidden_num = len(self.hidden_var)

        viterbi_mat = [[{} for _ in xrange(hidden_num)] for _ in xrange(seq_len)]

        observation = seq[0]
        for i in xrange(hidden_num):
            h_var = self.hidden_var[i]
            viterbi_mat[0][i][h_var] = self.initial_prob[h_var] * self.emission_prob[h_var][observation]

        for j in xrange(1, seq_len):
            observation = seq[j]  # the Jth observation
            for i in xrange(hidden_num):
                current_var = self.hidden_var[i]  # which hidden variable
                e_prob = self.emission_prob[current_var][observation]

                max_prob = 0
                var = 0
                for h in xrange(hidden_num):
                    best_prob = viterbi_mat[j - 1][h].values()[0]
                    previous_var = self.hidden_var[h]

                    current_prob = best_prob * self.transition_prob[previous_var][current_var] * e_prob
                    if current_prob > max_prob:
                        max_prob = current_prob
                        var = previous_var
                viterbi_mat[j][i][var] = max_prob

        hidden_seq = []
        max_prob = 0  # get the final latent variable
        hidden_var = 0
        for d in viterbi_mat[-1]:
            for k, v in d.iteritems():
                if v > max_prob:
                    hidden_var = k
        hidden_seq.append(hidden_var)

        for i in xrange(1, seq_len):
            hidden_var = viterbi_mat[-i][hidden_var - 1].keys()[0]
            hidden_seq.append(hidden_var)
        hidden_seq.reverse()
        return hidden_seq

    def get_hidden_sequence(self):
        return self.hidden_seq

    def get_observation_sequence(self):
        return self.observation_seq


def sampling(prob_dict):
    value = []
    prob = []
    for k, v in prob_dict.iteritems():
        value.append(k)
        prob.append(v)
    result = np.random.choice(value, 1, p=prob).tolist()[0]
    return result


def main():
    init_prob = {1: 0.6, 2: 0.4}
    trans_prob = {1: {1: 0.7, 2: 0.3}, 2: {1: 0.4, 2: 0.6}}
    emit_prob = {1: {1: 0.1, 2: 0.4, 3: 0.5}, 2: {1: 0.6, 2: 0.3, 3: 0.1}}

    hmm = HMM(init_prob, trans_prob, emit_prob)
    hmm.sample_sequence(10)
    o_result = hmm.get_observation_sequence()
    h_result = hmm.get_hidden_sequence()
    result = hmm.decoding([3, 2, 3, 2, 2, 3, 1, 1, 3, 3])
    print "The observation sequence is " + str(o_result)
    print "The corresponding hidden sequence is " + str(h_result)
    print "The decoding result of Viterbi algorithm is " + str(result)


if __name__ == "__main__":
    main()
