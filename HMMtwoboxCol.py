import numpy as np
from boxtask_func import *

a0 = 0  # a0 = do nothing
g0 = 1  # g0 = go to location 0
g1 = 2  # g1 = go toward box 1 (via location 0 if from 2)
g2 = 3  # g2 = go toward box 2 (via location 0 if from 1)
pb = 4  # pb  = push button


class HMMtwoboxCol:
    def __init__(self, A, B, C, D1, D2, pi, Ncol):
        self.A = A
        self.B = B
        self.C = C  # (Trans_hybrid_obs12, belief transition given color and action)
        self.D1 = D1  # (box1, Obs_emis.dot(Trans_state, to calculate oberservation emission)
        self.D2 = D2  # (box2, Obs_emis.dot(Trans_state, to calculate oberservation emission)
        self.pi = pi
        self.S = len(self.pi)  # number of possible values of the hidden state (hybrid for two boxes)
        self.R = 2
        self.L = 3
        self.Ss = int(sqrt(self.S))
        self.Ncol = Ncol

    def _states(self, r, l):
        temp = np.reshape(np.array(range(self.Ss)), [1, self.Ss])
        return np.squeeze(l * self.S * self.R + tensorsum(temp * self.R * self.Ss, r * self.Ss + temp)).astype(int)

    def forward_scale(self, obs):

        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]  # observable, two possible values: 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        alpha = np.zeros((self.S, T))  # initialize alpha value for each belief value
        scale = np.zeros(T)

        alpha[:, 0] = self.pi * self.B[act[0], self._states(rew[0], loc[0])]
        scale[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / scale[0]

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.Ss) + 1 / self.Ss / 2, 1 - np.arange(0, 1, 1 / self.Ss) - 1 / self.Ss / 2])

        for t in range(1, T):
            alpha[:, t] = np.dot(alpha[:, t - 1],
                                 self.C[col1[t]][col2[t]][act[t - 1]][
                                     np.ix_(self._states(rew[t - 1], loc[t - 1]), self._states(rew[t], loc[t]))]) \
                          * self.B[act[t], self._states(rew[t], loc[t])]

            scale[t] = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / scale[t]

        return alpha, scale

    def backward_scale(self, obs, scale):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        beta = np.zeros((self.S, T))
        beta[:, T - 1] = 1

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.Ss) + 1 / self.Ss / 2, 1 - np.arange(0, 1, 1 / self.Ss) - 1 / self.Ss / 2])

        for t in reversed(range(T - 1)):
            beta[:, t] = np.dot(self.C[col1[t + 1]][col2[t + 1]][act[t]][np.ix_(self._states(rew[t], loc[t]),
                                                                                self._states(rew[t + 1],
                                                                                             loc[t + 1]))],
                                beta[:, t + 1] * self.B[act[t + 1], self._states(rew[t + 1], loc[t + 1])])


            beta[:, t] = beta[:, t] / scale[t + 1]

        return beta

    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, 0)

        return gamma

    def compute_xi(self, alpha, beta, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        xi = np.zeros((T - 1, self.S, self.S))

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.Ss) + 1 / self.Ss / 2, 1 - np.arange(0, 1, 1 / self.Ss) - 1 / self.Ss / 2])

        for t in range(T - 1):
            xi[t, :, :] = np.diag(alpha[:, t]).dot(
                self.C[col1[t + 1]][col2[t + 1]][act[t]][
                    np.ix_(self._states(rew[t], loc[t]), self._states(rew[t + 1], loc[t + 1]))]
            ).dot(np.diag(beta[:, t + 1] * self.B[act[t + 1], self._states(rew[t + 1], loc[t + 1])]))


            xi[t, :, :] = xi[t, :, :] / np.sum(xi[t, :, :])

        return xi

    def latent_entr(self, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        # Entropy of all path that leads to a certain state at t certain time
        Hpath = np.zeros((self.S, T))
        # P(state at time t-1 | state at time t, observations up to time t)
        lat_cond = np.zeros((T - 1, self.S, self.S))

        alpha_scaled, _ = self.forward_scale(obs)
        Hpath[:, 0] = 0

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.Ss) + 1 / self.Ss / 2, 1 - np.arange(0, 1, 1 / self.Ss) - 1 / self.Ss / 2])

        for t in range(1, T):
            lat_cond[t - 1] = np.diag(alpha_scaled[:, t - 1]).dot(
                self.C[col1[t]][col2[t]][act[t - 1]][
                    np.ix_(self._states(rew[t - 1], loc[t - 1]), self._states(rew[t], loc[t]))])

            lat_cond[t - 1] = lat_cond[t - 1] / (
                    np.sum(lat_cond[t - 1], axis=0) + 1 * (np.sum(lat_cond[t - 1], axis=0) == 0))

            Hpath[:, t] = Hpath[:, t - 1].dot(lat_cond[t - 1]) - np.sum(
                lat_cond[t - 1] * np.log(lat_cond[t - 1] + 10 ** -13 * (lat_cond[t - 1] == 0)), axis=0)

        lat_ent = np.sum(Hpath[:, -1] * alpha_scaled[:, -1]) - np.sum(
            alpha_scaled[:, -1] * np.log(alpha_scaled[:, -1] + 10 ** -13 * (alpha_scaled[:, -1] == 0)))

        return lat_ent

    # def likelihood(self, lat, obs, Anew, Bnew):

    def computeQaux(self, obs, Anew, Bnew, Cnew, D1new, D2new):

        '''
        computer the Q auxillary funciton, the expected complete data likelihood
        :param obs: observations
        :param Anew: transition matrix with new parameters
        :param Bnew: policy with new parameters
        :param Cnew: Trans_hybrid_obs12 with new parameters
        :param D1new: box1, Obs_emis.dot(Trans_state, to calculate observation emission, with new parameters
        :param D2new: box2, Obs_emis.dot(Trans_state, to calculate observation emission, with new parameters
        :return: Q auxilary value
        '''
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        alpha, scale = self.forward_scale(obs)
        beta = self.backward_scale(obs, scale)

        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta, obs)
        Qaux1 = np.sum(np.log(self.pi) * gamma[:, 0])
        Qaux2 = 0
        Qaux3 = 0


        for t in range(T - 1):
            Trantemp = Cnew[col1[t + 1]][col2[t + 1]][act[t]][
                np.ix_(self._states(rew[t], loc[t]), self._states(rew[t + 1], loc[t + 1]))]

            Qaux2 += np.sum(np.log(Trantemp + 10 ** -13 * (Trantemp == 0)) * xi[t, :, :])

        for t in range(T):
            Qaux3 += np.sum(np.log(Bnew[act[t], self._states(rew[t], loc[t])] +
                                   10 ** -13 * (Bnew[act[t], self._states(rew[t], loc[t])] == 0)) * gamma[:, t])



        Qaux = 1 * (Qaux1 + Qaux2) + 1 * Qaux3

        return Qaux