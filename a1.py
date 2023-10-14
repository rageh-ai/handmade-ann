import numpy as np
import pandas as pd
import scipy.stats as stats
import sys


def sigmoid(x):

    return (1/(1+np.exp(-x)))


def passthrough(weights, inputs, bias):
    weighted_sum = np.dot(weights, inputs) + bias
    return (sigmoid(weighted_sum), weighted_sum)


def backprop(weight1, weight2, bias1, bias2, input_x, true):
    lr_rate = 0.7

    pass_1 = passthrough(weight1, input_x, bias1)
    pass_2 = passthrough(weight2, pass_1[0], bias2)

    loss_final = np.multiply((pass_2[0] - true), sigmoid_derivative((pass_2[1])))
    w_2_grad = loss_final * pass_1[0].T


    loss_l_1 = np.multiply(np.dot(weight2.T,loss_final), sigmoid_derivative(pass_1[1]))
    w_1_grad = loss_l_1 * input_x.T
    
    b_2_grad = loss_final
    b_1_grad = loss_l_1


    w1_new = weight1 - np.multiply(lr_rate,w_1_grad)
    w2_new = weight2 - np.multiply(lr_rate,w_2_grad)

    b1_new = bias1 - np.multiply(lr_rate, b_1_grad)
    b2_new = bias2 - np.multiply(lr_rate, b_2_grad)

    return w1_new, w2_new, b1_new, b2_new



def sigmoid_derivative(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def train_network(X):
    
    w_1 = np.random.randn(hidden_no,input_no) * 0.1
    w_2 = np.random.randn(output_no, hidden_no) * 0.1 

    b_1 = np.zeros((hidden_no,1))
    b_2 = np.zeros((output_no,1))

    for epoch_count in range(max_epochs):
        shuffled = X.sample(frac=1)
        for instance in range(len(shuffled)):
            #input instance
            x_d = shuffled.iloc[instance, :input_no].to_numpy()
            x_d = x_d.reshape(input_no,1)

            #getting true value
            true_val = shuffled.iloc[instance, input_no:].to_numpy().reshape(output_no,1)

            #backprop
            updates = backprop(w_1, w_2, b_1, b_2, x_d, true_val)

            w_1 = updates[0]
            w_2 = updates[1]
            b_1 = updates[2]
            b_2 = updates[3]
            
    return w_1, w_2, b_1, b_2



def test_network(df_test, parameters):
    test_out = np.empty((0,output_no))
    

    for example in range(len(df_test)):
        test_ex = df_test.iloc[example,:input_no].to_numpy()
        test_ex = test_ex.reshape(input_no,1)
        l_1_out = passthrough(parameters[0], test_ex, parameters[2])
        final_val = passthrough(parameters[1], l_1_out[0],parameters[3])[0]
        final_val = final_val.reshape(1, output_no)
        test_out = np.vstack([test_out, final_val])
    test_out = pd.DataFrame((test_out >= 0.5) * 1)
    test_out.to_csv("B00833926.csv", header=None, index=False)





def normalization(X):
    for column in X.columns:
        X[column] = stats.zscore(X[column])
    return X


if __name__ == "__main__":

    if len(sys.argv) != 7:

        print("Arguments must be in the following format:\
number_of_input_features number_of_outputs_features\
number_of hidden_layer_neurons maximum_number_of_passes_through_the_training_partition\
train_file_path test_file_path")
        sys.exit()
    
    input_no = int(sys.argv[1])
    output_no = int(sys.argv[2])
    hidden_no = int(sys.argv[3])
    max_epochs = int(sys.argv[4])
    train =  pd.read_csv(sys.argv[5])
    test =  pd.read_csv(sys.argv[6])


    #normalizing features 
    train.iloc[:, : input_no] = normalization(train.iloc[:, : input_no])
    test_res = test.iloc[:, input_no : ]
    test_res.to_csv("results.csv", header=None, index=False)
    test = normalization(test.iloc[:, : input_no])

    #training network
    parameters = train_network(train)

    #testing network with learned paramters
    test_network(test, parameters)
