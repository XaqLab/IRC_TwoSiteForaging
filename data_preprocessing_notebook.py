import pickle
import numpy as np
from pandas import DataFrame, read_csv
import os


"""
all data should be located in data folder.
all output data also will be located in data folder.

"""

def data_preprocessing_encoding( idx,datestring_IRC,
                                 datestring_train,
                                 datestring_data,
                                 datestring_NNagent,NEURAL_NUM = 100
                                 ):

    path = os.getcwd()

    dataN_pkl_file_agent = open(path + '/Results/' + datestring_train + '_data' + datestring_data +
                           '_agentNNdriven' + datestring_NNagent + '_twoboxCol.pkl', 'rb')

    dataN_pkl_file_IRC = open(path + '/Results/' + datestring_train + '_data' + datestring_data +
                              '_agentNN' + datestring_NNagent + '_' + datestring_IRC + 'IRC' + str(idx) + '_twoboxCol' + '.pkl', 'rb')

    dataN_pkl_agent = pickle.load(dataN_pkl_file_agent)
    dataN_pkl_file_agent.close()

    dataN_pkl_IRC = pickle.load(dataN_pkl_file_IRC)
    dataN_pkl_file_IRC.close()

    bbelief = np.dstack([dataN_pkl_IRC['belief1_est_MAP'], dataN_pkl_IRC['belief2_est_MAP']])
    # behavior belief, 1x1000x2, and here 2 means belief for two boxes.
    bb = bbelief.reshape(-1, 2)  # 2 beliefs for two boxes

    neural_response = dataN_pkl_agent['neural_response'][0:1]  # neural response
    r = neural_response.reshape(-1, NEURAL_NUM)  # NEURAL_NUM neurons

    bb_df = DataFrame(bb, columns=['behavior_belief1', 'behavior_belief2'])
    bb_df.to_csv(path_or_buf='./data/bb_df.csv', index=False)

    r_df = DataFrame(r)  # no column name
    r_df.to_csv(path_or_buf='./data/r_df.csv', index=False)

    print('data preprocessing for encoding is successfully done!')


def data_preprocessing_decoding_recodong(idx,datestring_IRC,
                                datestring_train,
                                datestring_data,
                                datestring_NNagent,
                                DECODING=True, RECODING=True, NEURAL_NUM=100
                                ):
    """
    Here 'dataN_pkl' is a then dictionary with keys: dict_keys(['observations', 'beliefs', 'trueStates', 'allData']).
    dataN_pkl['observations'] has shape 200x500x5, where 200 is the number of sequences, 500 is the length of one sequence, 5 means it contains action, reward, location, color of box1, color of box2.
    dataN_pkl['beliefs'] has shape 200x500x2, and here 2 means belief for two boxes.
    dataN_pkl['trueState'] is the binary true state information for each box, and dataN_pkl['allData'] is just a stack of the variables above.
    """


    path = os.getcwd()
    dataN_pkl_file_agent = open(path + '/Results/' + datestring_train + '_data' + datestring_data +
                                '_agentNNdriven' + datestring_NNagent + '_twoboxCol.pkl', 'rb')

    dataN_pkl_file_IRC = open(path + '/Results/' + datestring_train + '_data' + datestring_data +
                              '_agentNN' + datestring_NNagent + '_' + datestring_IRC + 'IRC' + str(idx) + '_twoboxCol' + '.pkl', 'rb')

    dataN_pkl_agent = pickle.load(dataN_pkl_file_agent)
    dataN_pkl_file_agent.close()
    obs = dataN_pkl_agent['observations'][idx:idx+1]

    dataN_pkl_IRC = pickle.load(dataN_pkl_file_IRC)
    dataN_pkl_file_IRC.close()

    bbelief = np.dstack([dataN_pkl_IRC['belief1_est_MAP'], dataN_pkl_IRC['belief2_est_MAP']])
    # behavior belief, 1x1000x2, and here 2 means belief for two boxes.
    bb = bbelief.reshape(-1, 2)  # 2 beliefs for two boxes

    neural_response = dataN_pkl_agent['neural_response'][0:1]  # neural response
    r = neural_response.reshape(-1, NEURAL_NUM)  # NEURAL_NUM neurons

    nbelief = read_csv('./data/nb_df.csv').to_numpy().reshape(bbelief.shape)
    # This should have the same shape as bbelief, and corresponds to estimation of bbelief
    nb = nbelief.reshape(-1, 2)

    if DECODING: # need belief, location
        action = obs[:, :, 0]  # actions
        a = action.reshape(-1, 1)  # one action
        location = obs[:, :, 2].reshape(-1, 1)  # location
        reward = obs[:, :, 1].reshape(-1, 1)  # reward
        policy = obs[:, :, 5:].squeeze()  #policy
        policy_POMDP = dataN_pkl_agent['POMDP_agent_dist'][idx:idx+1, :, :5].squeeze()
        belief_POMDP = dataN_pkl_agent['POMDP_agent'][idx:idx+1, :, 1:3].squeeze()

        decoding_data = np.concatenate((nb, a, location, reward, policy, policy_POMDP, belief_POMDP), axis=1)
        decoding_data_df = DataFrame(decoding_data, columns=['neural_belief1', 'neural_belief2', 'action',
                                                             'location','reward', 'p1', 'p2', 'p3', 'p4', 'p5',
                                                             'POMDP_p1', 'POMDP_p2', 'POMDP_p3', 'POMDP_p4', 'POMDP_p5',
                                                             'POMDP_belief1', 'POMDP_belief2'])
        decoding_data_df.to_csv(path_or_buf='./data/neural_decoding_data.csv', index=False)



    if RECODING:
        """
        To build a data for recoding
        """
        obs_index = [0,2,3,4] #action, location, color 1, color 2
        nb_prev = nbelief[0, :-1, :]
        nb_now = nbelief[0, 1:, :]
        obs_prev = obs[0, :-1, obs_index].T
        obs_now = obs[0, 1:, obs_index].T

        if nbelief.shape[0] > 1:
            for i in range(1, nbelief.shape[0]):  # start from 2 on purpose (the first 500 data are useless - skip 0)
                # print(i)
                nb_prev = np.concatenate((nb_prev, nbelief[i, :-1, :]), axis=0)
                nb_now = np.concatenate((nb_now, nbelief[i, 1:, :]), axis=0)
                obs_prev = np.concatenate((obs_prev, obs[i, :-1, obs_index].T), axis=0)
                obs_now = np.concatenate((obs_now, obs[i, 1:, obs_index].T), axis=0)

        recoding_data_prev = np.concatenate((nb_prev, obs_prev), axis=1)
        recoding_data_now = np.concatenate((nb_now, obs_now), axis=1)

        recoding_data_prev_df = DataFrame(recoding_data_prev,
                                          columns=['neural_belief1', 'neural_belief2', 'action',
                                                   'location', 'color 1',
                                                   'color 2'])
        recoding_data_now_df = DataFrame(recoding_data_now,
                                         columns=['neural_belief1', 'neural_belief2', 'action',
                                                  'location', 'color 1',
                                                  'color 2'])
        recoding_data_prev_df.to_csv(path_or_buf='./data/recoding_neural_all_prev_df.csv', index=False)
        recoding_data_now_df.to_csv(path_or_buf='./data/recoding_neural_all_now_df.csv', index=False)

        # Also save the data for b_hat from IRC, for the purpose of  comparing dynamic functions
        bb_prev = bbelief[0, :-1, :]
        bb_now = bbelief[0, 1:, :]
        obs_prev = obs[0, :-1, obs_index].T
        obs_now = obs[0, 1:, obs_index].T

        if bbelief.shape[0] > 1:
            for i in range(1, bbelief.shape[0]):  # start from 2 on purpose (the first 500 data are useless - skip 0)
                # print(i)
                bb_prev = np.concatenate((bb_prev, bbelief[i, :-1, :]), axis=0)
                bb_now = np.concatenate((bb_now, bbelief[i, 1:, :]), axis=0)
                obs_prev = np.concatenate((obs_prev, obs[i, :-1, obs_index].T), axis=0)
                obs_now = np.concatenate((obs_now, obs[i, 1:, obs_index].T), axis=0)

        recoding_data_prev_IRC = np.concatenate((bb_prev, obs_prev), axis=1)
        recoding_data_now_IRC = np.concatenate((bb_now, obs_now), axis=1)

        recoding_data_prev_df_IRC = DataFrame(recoding_data_prev_IRC,
                                          columns=['behavior_belief1', 'behavior_belief2', 'action',
                                                   'location', 'color 1',
                                                   'color 2'])
        recoding_data_now_df_IRC = DataFrame(recoding_data_now_IRC,
                                         columns=['behavior_belief1', 'behavior_belief2', 'action',
                                                  'location', 'color 1',
                                                  'color 2'])
        recoding_data_prev_df_IRC.to_csv(path_or_buf='./data/recoding_IRC_all_prev_df.csv', index=False)
        recoding_data_now_df_IRC.to_csv(path_or_buf='./data/recoding_IRC_all_now_df.csv', index=False)

        print('data preprocessing for recoding/decoding is successfully done!')





