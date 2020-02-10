"""
######################################################
-- RNN model class --
Architecture:
Input layer --> Hidden recurrent layer (hidden_size1)
                       --> Linear with RELU (hidden_size2)
                                  --> Linear with softmax (output_size_act = na)-- > action

#####################################################
"""
from POMDP_generate import *
from twoCol_NN_data_utils import *
from twoCol_NN_model_generalization import *

def training_generalization(parameterSetFull, xMatFull, yMatFull, POMDP_params, training_params, nn_params, datestring_data, datestring_train):
    input_size, hidden_size1, hidden_size2, output_size_act, num_layers = nn_params
    batch_size, train_ratio, NEpochs, lr = training_params
    nq, na, nr, nl, Numcol, discount, parametersAgent_set = POMDP_params

    # save neural network and training parameters
    nn_train_para_dict = {'POMDP_params': POMDP_params,
                          'nn_params': nn_params,
                          'training_params': training_params
                          }
    nn_train_para_output = open(
        path + '/Results/' + datestring_train + '_data' + datestring_data +
        '_nn_train_params_twoboxCol_generalization.pkl', 'wb')
    pickle.dump(nn_train_para_dict, nn_train_para_output)
    nn_train_para_output.close()

    dataset = data_utils.TensorDataset(torch.tensor(parameterSetFull, dtype=torch.float),
                                       torch.tensor(xMatFull, dtype=torch.float),
                                       torch.tensor(yMatFull, dtype=torch.float))

    Ns = xMatFull.shape[0]
    train_set, test_set = data_utils.random_split(dataset, [int(Ns * train_ratio), Ns - int(Ns * train_ratio)])
    train_loader = data_utils.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test_set, batch_size)

    """
    Create RNN module
    """
    rnn = RNN(input_size, hidden_size1, hidden_size2, output_size_act, num_layers, parameterSetFull.shape[1])
    criterion_act = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer_act = torch.optim.Adam(rnn.parameters(), lr)

    train_loss = np.zeros([NEpochs])

    for epoch in range(NEpochs):
        for i, data in enumerate(train_loader, 0):
            para_batch, in_batch, target_batch = data
            target_bel_batch = target_batch[:, :, 0:2]
            target_act_batch = target_batch[:, :, 2]
            target_actDist_batch = target_batch[:, :, 3:]

            out_act_batch, hidden_batch = rnn(in_batch, para_batch)

            loss = criterion_act(torch.log(out_act_batch), target_actDist_batch)
            # loss = criterion(out_bel_batch.squeeze(), target_batch[:, :, 0:2])

            optimizer_act.zero_grad()  # zero-gradients at the start of each epoch
            loss.backward()
            optimizer_act.step()

        train_loss[epoch] = loss.item()

        if epoch % 10 == 9:
            print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))

            torch.save({
                'epoch': epoch,
                'model_state_dict': rnn.state_dict(),
                'optimizer_state_dict': optimizer_act.state_dict(),
                'loss': loss,
            }, path + '/Results/' + datestring_train + '_rnn' + '_epoch' + str(
                epoch + 1) + '_dataEnsemble' + datestring_data)

    print("Learning finished!")

    return rnn, test_loader, train_loader, train_loss






