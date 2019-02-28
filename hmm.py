import numpy as np

class HMM(object):
    """
    一阶隐马尔可夫模型
    A: 转移概率矩阵（每一行代表上一个隐状态）
    B：发射矩阵（每一行代表一个隐状态）
    pi:初始概率
    """
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def _forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((N, T))
        F[:, 0] = self.pi*self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.dot(F[:, t-1], self.A[:, n])*self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N, T))
        X[:, -1] = 1

        for t in reversed(range(T-1)):
            for n in range(N):
                X[n, t] = np.sum(X[:, t+1]*self.A[n,:]*self.B[:, obs_seq[t+1]])

        return X

    def observation_prob(self, obs_seq):
        """
        输出特定观察序列下的概率
        """
        return np.sum(self._forward(obs_seq)[:, -1])

    def state_path(self, obs_seq):
        """
        Returns:
        ----------
        V[last_state, -1] : float
        prob of the optimal state path
        path: list(int)
        the optimal path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        last_state =np.argmax(V[:, -1])
        path = list(self.bulid_viteri_path(prev, last_state))

        return V[last_state, -1], reversed(path)

    def viterbi(self, obs_seq):
        """
        Returns:
        -----------
        V[s][t]: Maximum prob of an observation sequence ending at time
                 't' with state 's'
        prev :a pointer to the previous state at t-1 maximizes
              V[state][t]
        """
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T-1, N), dtype=int)

        V = np.zeros((N, T))
        V[:, 0] = self.pi*self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                seq_probs = V[:, t-1]*self.A[:, n]*self.B[n, obs_seq[t]]
                prev[t-1, n] = np.argmax(seq_probs)
                V[n, t] = np.max(seq_probs)

        return V, prev

    def bulid_viteri_path(self, prev, last_state):
        """
        Returns: return a state path ending in last_state in a reversed list
        """
        T = len(prev)
        yield(last_state)
        for i in range(T-1, -1, -1):
            yield(prev[i, last_state])
            last_state = prev[i, last_state]

    def simulate(self, T):
        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0], :])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t-1], :])
            observations[t] = draw_from(self.B[states[t], :])
        return observations, states

    def baum_welch_train(self, observations, criterion=0.05):
        n_states = self.A.shape[0]
        n_samples = len(observations)

        done = False
        while not done:
            # alpha_t(i) = P(o_1, o_2, ......, o_t, q_t = S_i|hmm)
            # init beta
            alpha = self._forward(observations)
            beta = self._backward(observations)

            xi = np.zeros((n_states, n_states, n_samples-1))
            for t in range(n_samples-1):
                denom = np.dot(np.dot(alpha[:, t].T, self.A)*self.B[:, observations[t+1]].T, beta[:, t+1])
                for i in range(n_states):
                    numer = alpha[i, t]* self.A[i, :]*self.B[:, observations[t+1]].T*beta[:, t+1].T
                    xi[i, :, t] = numer/denom
            # gamma_t(i) = P(q_t = S_i|O, hmm)
            gamma = np.squeeze(np.sum(xi, axis=1))
            prod = (alpha[:, n_samples-1]*beta[:, n_samples-1]).reshape((-1, 1))
            gamma = np.hstack((gamma, prod/np.sum(prod)))

            newpi = gamma[:, 0]
            newA = np.sum(xi, 2)/np.sum(gamma[:,:-1], axis=1).reshape((-1, 1))
            newB = np.copy(self.B)

            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma, axis=1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:, lev] = np.sum(gamma[:, mask], axis=1)/sumgamma

            if np.max(abs(self.pi - newpi)) < criterion and\
               np.max(abs(self.A - newA)) < criterion and\
               np.max(abs(self.B - newB)) < criterion:
               done = True

            self.A[:], self.B[:], self.pi[:] = newA, newB, newpi
