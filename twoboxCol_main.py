from twoboxCol import *
from HMMtwoboxCol import *
import pickle
import sys
from datetime import datetime
import os
path = os.getcwd()

E_MAX_ITER = 300       # 100    # maximum number of iterations of E-step
GD_THRESHOLD = 0.01   # 0.01      # stopping criteria of M-step (gradient descent)
E_EPS = 10 ** -2                  # stopping criteria of E-step
M_LR_INI = 5  * 10 ** -5           # initial learning rate in the gradient descent step
LR_DEC =  4                       # number of times that the learning rate can be reduced
SaveEvery = 10

def twoboxColGenerate(parameters, parametersExp, sample_length, sample_number, nq, nr = 2, nl = 3, na = 5, discount = 0.99):
    # datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')  # current time used to set file name

    print("\nSet the parameters of the model... \n")

    ### Set all related parameters
    #     beta = parameters[0]
    #     gamma1 = parameters[1]
    #     gamma2 = parameters[2]
    #     delta =  parameters[3]
    #     direct = parameters[4]
    #     epsilon1 = parameters[5]
    #     epsilon2 = parameters[6]
    #     rho = parameters[7]
    #     # State rewards
    #     Reward = 1
    #     groom = parameters[8]
    #     # Action costs
    #     travelCost = parameters[9]
    #     pushButtonCost = parameters[10]

    beta = 0  # available food dropped back into box after button press
    gamma1 = parameters[0]  # reward becomes available in box 1
    gamma2 = parameters[1]  # reward becomes available in box 2
    delta = 0  # animal trips, doesn't go to target location
    direct = 0  # animal goes right to target, skipping location 0
    epsilon1 = parameters[2]  # available food disappears from box 1
    epsilon2 = parameters[3]  # available food disappears from box 2
    rho = 1  # food in mouth is consumed
    # State rewards
    Reward = 1  # reward per time step with food in mouth
    groom = parameters[4]  # location 0 reward
    # Action costs
    travelCost = parameters[5]
    pushButtonCost = parameters[6]

    NumCol = np.rint(parameters[7]).astype(int)  # number of colors
    Ncol = NumCol - 1  # max value of color
    qmin = parameters[8]
    qmax = parameters[9]

    gamma1_e = parametersExp[0]
    gamma2_e = parametersExp[1]
    epsilon1_e = parametersExp[2]
    epsilon2_e = parametersExp[3]
    qmin_e = parametersExp[4]
    qmax_e = parametersExp[5]

    # parameters = [gamma1, gamma2, epsilon1, epsilon2,
    #              groom, travelCost, pushButtonCost, NumCol, qmin, qmax]

    ### Gnerate data"""
    print("Generating data...")
    T = sample_length
    N = sample_number
    twoboxColdata = twoboxColMDPdata(discount, nq, nr, na, nl, parameters, parametersExp, T, N)
    twoboxColdata.dataGenerate_sfm(belief1Initial=0, rewInitial=0, belief2Initial=0, locationInitial=0)
    # twoboxdata.dataGenerate_op(belief1Initial=0, rewInitial=0, belief2Initial=0, locationInitial=0)

    hybrid = twoboxColdata.hybrid
    action = twoboxColdata.action
    location = twoboxColdata.location
    belief1 = twoboxColdata.belief1
    belief2 = twoboxColdata.belief2
    reward = twoboxColdata.reward
    trueState1 = twoboxColdata.trueState1
    trueState2 = twoboxColdata.trueState2
    color1 = twoboxColdata.color1
    color2 = twoboxColdata.color2

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = np.dstack([action, reward, location, color1, color2])  # includes the action and the observable states
    latN = np.dstack([belief1, belief2])
    truthN = np.dstack([trueState1, trueState2])
    dataN = np.dstack([obsN, latN, truthN])

    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}
    data_output = open(datestring + '_dataN_twoboxCol' + '.pkl', 'wb')
    pickle.dump(data_dict, data_output)
    data_output.close()

    ### write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'nl': nl,
                 'na': na,
                 'foodDrop': beta,
                 'appRate1': gamma1,
                 'appRate2': gamma2,
                 'disappRate1': epsilon1,
                 'disappRate2': epsilon2,
                 'consume': rho,
                 'reward': Reward,
                 'groom': groom,
                 'travelCost': travelCost,
                 'pushButtonCost': pushButtonCost,
                 'ColorNumber': NumCol,
                 'qmin': qmin,
                 'qmax': qmax,
                 'appRateExperiment1': gamma1_e,
                 'disappRateExperiment1': epsilon1_e,
                 'appRateExperiment2': gamma2_e,
                 'disappRateExperiment2': epsilon2_e,
                 'qminExperiment': qmin_e,
                 'qmaxExperiment': qmax_e
                 }

    # create a file that saves the parameter dictionary using pickle
    para_output = open(datestring + '_para_twoboxCol' + '.pkl', 'wb')
    pickle.dump(para_dict, para_output)
    para_output.close()

    print('Data stored in files')

    return obsN, latN, truthN, datestring

def main():
    ##############################################
    #
    #   python -u twoboxCol_main.py [0.35,0.3,0.15,0.1,0.1,0.3,0.6,5,0.42,0.66] [0.15,0.3,0.1,0.15,0.4,0.6] \([0.4,0.3,0.19,0.2,0.16,0.25,0.5,5,0.3,0.7]-[0.2,0.4,0.1,0.25,0.12,0.2,0.4,5,0.5,0.6]\) > $(date +%m%d%Y\(%H%M\))_twoboxCol.txt &

    #
    ##############################################

    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')  # current time used to set file name

    # parameters = [gamma1, gamma2, epsilon1, epsilon2, groom, travelCost, pushButtonCost, NumCol, qmin, qmax]
    parametersAgent = np.array(list(map(float, sys.argv[1].strip('[]').split(','))))
    parametersExp = np.array(list(map(float, sys.argv[2].strip('[]').split(','))))

    #parameters_gen = [0.1,0.1,0.01,0.01,0.05,0.2,0.3,5,0.4,0.6]

    #obsN, latN, truthN, datestring = twoboxColGenerate(parametersAgent, parametersExp, sample_length = 5000, sample_number = 1, nq = 5)
    # sys.stdout = logger.Logger(datestring)
    # output will be both on the screen and in the log file
    # No need to manual interaction to specify parameters in the command line
    # but the log file will be written at the end of the execution.

    dataN_pkl_file = open(path + '/Results/04302019(0246)_dataN_twoboxCol.pkl', 'rb')
    dataN_pkl = pickle.load(dataN_pkl_file)
    dataN_pkl_file.close()
    obsN = dataN_pkl['observations']
    latN = dataN_pkl['beliefs']

    ######## Hyper-parameters ###############################
    parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
                          'GD_THRESHOLD': GD_THRESHOLD,
                          'E_EPS': E_EPS,
                          'M_LR_INI': M_LR_INI,
                          'LR_DEC': LR_DEC,
                          'ParaInitial': [np.array(list(map(float, i.strip('[]').split(',')))) for i in
                                          sys.argv[3].strip('()').split('-')]
                          #'ParaInitial': [np.array(list(map(float, i.strip('[]').split(',')))) for i in
                          #                sys.argv[3].strip('()').split('-')]
                          # Initial parameter is a set that contains arrays of parameters, here only consider one initial point
                          }

    output1 = open(datestring + '_ParameterMain_twoboxCol' + '.pkl', 'wb')
    pickle.dump(parameterMain_dict, output1)
    output1.close()
    ##############################################################


    ### Choose which sample is used for inference
    sampleIndex = [0]
    NN = len(sampleIndex)

    ### Set initial parameter point
    parameters_iniSet = parameterMain_dict['ParaInitial']

    ### read real para from data file
    #pkl_parafile = open(datestring + '_para_twoboxCol' + '.pkl', 'rb')
    pkl_parafile = open(path + '/Results/04302019(0246)_para_twoboxCol' + '.pkl', 'rb')
    para_pkl = pickle.load(pkl_parafile)
    pkl_parafile.close()

    discount = para_pkl['discount']
    nq = para_pkl['nq']
    nr = para_pkl['nr']
    nl = para_pkl['nl']
    na = para_pkl['na']
    beta = para_pkl['foodDrop']
    gamma1 = para_pkl['appRate1']
    gamma2 = para_pkl['appRate2']
    epsilon1 = para_pkl['disappRate1']
    epsilon2 = para_pkl['disappRate1']
    rho = para_pkl['consume']
    Reward = para_pkl['reward']
    groom = para_pkl['groom']
    travelCost = para_pkl['travelCost']
    pushButtonCost = para_pkl['pushButtonCost']
    NumCol = para_pkl['ColorNumber']
    qmin = para_pkl['qmin']
    qmax = para_pkl['qmax']
    gamma1_e = para_pkl['appRateExperiment1']
    epsilon1_e = para_pkl['disappRateExperiment1']
    gamma2_e = para_pkl['appRateExperiment2']
    epsilon2_e = para_pkl['disappRateExperiment2']
    qmin_e = para_pkl['qminExperiment']
    qmax_e = para_pkl['qmaxExperiment']

    print("\nThe true world parameters of box1 are:", "appearing rate =",
          gamma1_e, ",disappearing rate =", epsilon1_e)
    print("The true world parameters of box2 are:", "appearing rate =",
          gamma2_e, ",disappearing rate =", epsilon2_e)
    print("\nThe color parameters are:" "qmin_e =", qmin_e, "and qmax_e =", qmax_e)

    parameters = [gamma1, gamma2, epsilon1, epsilon2,
                  groom, travelCost, pushButtonCost,
                  NumCol, qmin, qmax]
    print("\nThe internal model parameters are", parameters)
    print("gamma1/2, rate that food appears of box 1/2"
          "\nepsilon1/2, rate that food disappears of box 1/2"
          "\ngroom, reward of grooming"
          "\ntravelCost, cost of traveling action"
          "\npushButtonCost, cost of pressing the button per unit of reward"
          "\nNcol, number of colors (assume equal to experiment setting)"
          "\nqmin, color parameter"
          "\nqmax, color parameter")

    print("\nThe initial points for estimation are:", parameters_iniSet)

    ### EM algorithm for parameter estimation
    print("\nEM algorithm begins ...")
    # NN denotes multiple data set, and MM denotes multiple initial points
    # NN_MM_para_old_traj = []
    # NN_MM_para_new_traj = []
    # NN_MM_log_likelihoods_old = []
    # NN_MM_log_likelihoods_new = []
    # NN_MM_log_likelihoods_com_old = []  # old posterior, old parameters
    # NN_MM_log_likelihoods_com_new = []  # old posterior, new parameters
    # NN_MM_latent_entropies = []

    for nn in range(NN):

        print("\nFor the", sampleIndex[nn] + 1, "-th set of data:")

        ##############################################################
        # Compute likelihood
        obs = obsN[sampleIndex[nn], :, :]

        MM = len(parameters_iniSet)

        # MM_para_old_traj = []
        # MM_para_new_traj = []
        # MM_log_likelihoods_old = []
        # MM_log_likelihoods_new = []
        # MM_log_likelihoods_com_old = []  # old posterior, old parameters
        # MM_log_likelihoods_com_new = []  # old posterior, new parameters
        # MM_latent_entropies = []

        for mm in range(MM):
            parameters_old = np.copy(parameters_iniSet[mm])

            print("\n######################################################\n",
                  mm + 1, "-th initial estimation:", parameters_old)

            itermax = E_MAX_ITER  # 100  # iteration number for the EM algorithm
            eps = E_EPS  # Stopping criteria for E-step in EM algorithm

            para_old_traj = []
            para_new_traj = []

            log_likelihoods_old = []
            log_likelihoods_new = []
            log_likelihoods_com_old = []  # old posterior, old parameters
            log_likelihoods_com_new = []  # old posterior, new parameters
            latent_entropies = []

            count_E = 0
            while True:

                print("\n The", count_E + 1, "-th iteration of the EM(G) algorithm")

                if count_E == 0:
                    parameters_old = np.copy(parameters_iniSet[mm])
                else:
                    parameters_old = np.copy(parameters_new)  # update parameters

                para_old_traj.append(parameters_old)

                ##########  E-step ##########

                ## Use old parameters to estimate posterior
                # twoboxGra = twoboxMDPder(discount, nq, nr, na, nl, parameters_old, vinitial)
                twoboxColGra = twoboxColMDPder(discount, nq, nr, na, nl, parameters_old)
                ThA_old = twoboxColGra.ThA
                softpolicy_old = twoboxColGra.softpolicy
                Trans_hybrid_obs12_old = twoboxColGra.Trans_hybrid_obs12
                Obs_emis_trans1_old = twoboxColGra.Obs_emis_trans1
                Obs_emis_trans2_old = twoboxColGra.Obs_emis_trans2
                Ncol_old = parameters_old[7].astype(int) - 1
                pi = np.ones(nq * nq) / nq / nq
                twoColHMM = HMMtwoboxCol(ThA_old, softpolicy_old, Trans_hybrid_obs12_old,
                                         Obs_emis_trans1_old, Obs_emis_trans2_old, pi, Ncol_old)

                ## Calculate likelihood of observed and complete date, and entropy of the latent sequence
                complete_likelihood_old = twoColHMM.computeQaux(obs, ThA_old, softpolicy_old,
                                                                Trans_hybrid_obs12_old, Obs_emis_trans1_old, Obs_emis_trans2_old)
                latent_entropy = twoColHMM.latent_entr(obs)
                log_likelihood = complete_likelihood_old + latent_entropy

                log_likelihoods_com_old.append(complete_likelihood_old)
                latent_entropies.append(latent_entropy)
                log_likelihoods_old.append(log_likelihood)

                print(parameters_old)
                print(complete_likelihood_old)
                print(log_likelihood)

                ## Check convergence


                ##########  M(G)-step ##########
                M_thresh = GD_THRESHOLD
                count_M = 0
                #vinitial = 0
                para_new_traj.append([])
                log_likelihoods_com_new.append([])
                log_likelihoods_new.append([])

                learnrate_ini = M_LR_INI
                learnrate = learnrate_ini

                # Start the gradient descent from the old parameters
                parameters_new = np.copy(parameters_old)
                complete_likelihood_new = complete_likelihood_old
                log_likelihood = complete_likelihood_new + latent_entropy

                para_new_traj[count_E].append(parameters_new)
                log_likelihoods_com_new[count_E].append(complete_likelihood_new)
                log_likelihoods_new[count_E].append(log_likelihood)

                print("\nM-step")
                print(parameters_new)
                print(complete_likelihood_new)
                print(log_likelihood)

                gra_sqrt = 0

                while True:

                    derivative_value = twoboxColGra.dQauxdpara_sim(obs, parameters_new)
                    print(derivative_value)
                    # vinitial is value from previous iteration, this is for computational efficiency
                    para_temp = parameters_new + learnrate * np.array(derivative_value)
                    #gra_sqrt = np.sqrt(0.9 * np.square(derivative_value) + 0.1 * gra_sqrt + 10 ** (-6))
                    #para_temp = parameters_new + learnrate * np.array(derivative_value) / gra_sqrt
                    #print(np.array(derivative_value) / gra_sqrt)
                    #para_temp[-3] = 5
                    # vinitial = derivative_value[-1]  # value iteration starts with value from previous iteration

                    ## Check the ECDLL (old posterior, new parameters)
                    twoboxCol_new = twoboxColMDP(discount, nq, nr, na, nl, para_temp)
                    twoboxCol_new.setupMDP()
                    twoboxCol_new.solveMDP_sfm()
                    ThA_new = twoboxCol_new.ThA
                    softpolicy_new = twoboxCol_new.softpolicy
                    Trans_hybrid_obs12_new = twoboxCol_new.Trans_hybrid_obs12
                    Obs_emis_trans1_new = twoboxCol_new.Obs_emis_trans1
                    Obs_emis_trans2_new = twoboxCol_new.Obs_emis_trans2
                    complete_likelihood_new_temp = twoColHMM.computeQaux(obs, ThA_new,softpolicy_new, Trans_hybrid_obs12_new,
                                                                         Obs_emis_trans1_new, Obs_emis_trans2_new)

                    print("         ", para_temp)
                    print("         ", complete_likelihood_new_temp)

                    ## Update the parameter if the ECDLL can be improved
                    if complete_likelihood_new_temp > complete_likelihood_new + M_thresh:
                        parameters_new = np.copy(para_temp)
                        complete_likelihood_new = complete_likelihood_new_temp
                        log_likelihood = complete_likelihood_new + latent_entropy

                        para_new_traj[count_E].append(parameters_new)
                        log_likelihoods_com_new[count_E].append(complete_likelihood_new)
                        log_likelihoods_new[count_E].append(log_likelihood)

                        print('\n', parameters_new)
                        print(complete_likelihood_new)
                        print(log_likelihood)

                        count_M += 1

                        # if count_M == 10 :
                        #    M_thresh = M_thresh / 4
                    else:
                        learnrate /= 2
                        if learnrate < learnrate_ini / (2 ** LR_DEC):
                            break

                # every 50 iterations, download data
                if (count_E + 1) % SaveEvery == 0:
                    Experiment_dict = {'ParameterTrajectory_Estep': para_old_traj,
                                       'ParameterTrajectory_Mstep': para_new_traj,
                                       'LogLikelihood_Estep': log_likelihoods_old,
                                       'LogLikelihood_Mstep': log_likelihoods_new,
                                       'Complete_LogLikelihood_Estep': log_likelihoods_com_old,
                                       'Complete_LogLikelihood_Mstep': log_likelihoods_com_new,
                                       'Latent_entropies': latent_entropies
                                       }
                    output = open(datestring + '_' + str(NN) + '_' + str(MM) + '_' + str(count_E + 1) + '_EM_twoboxCol' + '.pkl', 'wb')
                    pickle.dump(Experiment_dict, output)
                    output.close()

                count_E += 1

            # save the remainings (The last one contains all the parameters in the trajectory)
            Experiment_dict = {'ParameterTrajectory_Estep': para_old_traj,
                               'ParameterTrajectory_Mstep': para_new_traj,
                               'LogLikelihood_Estep': log_likelihoods_old,
                               'LogLikelihood_Mstep': log_likelihoods_new,
                               'Complete_LogLikelihood_Estep': log_likelihoods_com_old,
                               'Complete_LogLikelihood_Mstep': log_likelihoods_com_new,
                               'Latent_entropies': latent_entropies
                               }
            output = open(datestring + '_' + str(NN) + '_' + str(MM) + '_' + str(count_E + 1) + '_EM_twoboxCol' + '.pkl', 'wb')
            pickle.dump(Experiment_dict, output)
            output.close()

        #     MM_para_old_traj.append(para_old_traj)  # parameter trajectories for a particular set of data
        #     MM_para_new_traj.append(para_new_traj)
        #     MM_log_likelihoods_old.append(log_likelihoods_old)  # likelihood trajectories for a particular set of data
        #     MM_log_likelihoods_new.append(log_likelihoods_new)
        #     MM_log_likelihoods_com_old.append(log_likelihoods_com_old)  # old posterior, old parameters
        #     MM_log_likelihoods_com_new.append(log_likelihoods_com_new)  # old posterior, new parameters
        #     MM_latent_entropies.append(latent_entropies)
        #
        # NN_MM_para_old_traj.append(MM_para_old_traj)  # parameter trajectories for all data
        # NN_MM_para_new_traj.append(MM_para_new_traj)
        # NN_MM_log_likelihoods_old.append(MM_log_likelihoods_old)  # likelihood trajectories for
        # NN_MM_log_likelihoods_new.append(MM_log_likelihoods_new)
        # NN_MM_log_likelihoods_com_old.append(MM_log_likelihoods_com_old)  # old posterior, old parameters
        # NN_MM_log_likelihoods_com_new.append(MM_log_likelihoods_com_new)  # old posterior, new parameters
        # NN_MM_latent_entropies.append(MM_latent_entropies)

    #### Save result data and outputs log

    ## save the running data
    # Experiment_dict = {'ParameterTrajectory_Estep': NN_MM_para_old_traj,
    #                    'ParameterTrajectory_Mstep': NN_MM_para_new_traj,
    #                    'LogLikelihood_Estep': NN_MM_log_likelihoods_old,
    #                    'LogLikelihood_Mstep': NN_MM_log_likelihoods_new,
    #                    'Complete_LogLikelihood_Estep': NN_MM_log_likelihoods_com_old,
    #                    'Complete_LogLikelihood_Mstep': NN_MM_log_likelihoods_com_new,
    #                    'Latent_entropies': NN_MM_latent_entropies
    #                    }
    # output = open(datestring + '_EM_twoboxCol' + '.pkl', 'wb')
    # pickle.dump(Experiment_dict, output)
    # output.close()



    print("finish")

if __name__ == "__main__":
    main()
