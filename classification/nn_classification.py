import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)
NUM_FEATS = 90
epoch_losses = []
dev_losses = []

def sigmoid(h, derivative=False):
    h = 1/(1+np.exp(-h))
    if derivative:
        h = h*(1-h)
    return h
                                       
def relu(h, derivative=False):        
    if derivative:
        return np.heaviside(h ,0)
    return np.maximum(0, h)

def leaky_relu(h, alpha=0.1, derivative=False):
    if derivative:
        return np.where(h>0, 1, alpha)
    return np.where(h>0, h, h*alpha)

def softmax(h, derivative=False):
    h = np.exp(h)/np.sum(np.exp(h), axis=1).reshape(h.shape[0],1)
    if derivative:
        return h*(1-h)
    return h

def stable_softmax(h, derivative=False):
    h = np.exp(h- h.max())/np.sum(np.exp(h-h.max()), axis=1).reshape(h.shape[0],1)
    if derivative:
        return h*(1-h)
    return h

class Net(object):
    '''
    '''

    def __init__(self, num_layers, num_units, num_classes):
        '''
        Initialize the neural network.
        Create weights and biases.
        Here, we have provided an example structure for the weights and biases.
        It is a list of weight and bias matrices, in which, the
        dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
        weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
        biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]
        Please note that this is just an example.
        You are free to modify or entirely ignore this initialization as per your need.
        Also you can add more state-tracking variables that might be useful to compute
        the gradients efficiently.
        Parameters
        ----------
            num_layers : Number of HIDDEN layers.
            num_units : Number of units in each Hidden layer.
        '''
        if len(num_units)!=num_layers:
            raise Exception('Number of Hidden Layers and list of neurons are not compatible')
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_classes = num_classes
        self.biases = []
        self.weights = []
        for i in range(num_layers):
            if i==0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units[i])))
            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units[i-1], self.num_units[i])))
            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units[i], 1)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(self.num_classes, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units[-1], self.num_classes)))

        self.h_states = []
        self.a_states = []
        self.pred = 0

    def __call__(self, X, activation_fn):
        '''
        Forward propagate the input X through the network,
        and return the output.
        Note that for a classification task, the output layer should
        be a softmax layer. So perform the computations accordingly
        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
        Returns
        ----------
            y : Output of the network, numpy array of shape m x num_classes
        '''
        self.h_states = []
        self.a_states = []
        a, h = X, X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.h_states.append(h)
            self.a_states.append(a)

            h = np.dot(a, w) + b.T
            a = softmax(h) if i==len(self.weights)-1 else activation_fn(h)

        self.h_states.append(h)
        self.a_states.append(a)

        self.pred = a
        return self.pred
#         raise NotImplementedError

    def backward(self, X, y, batch_size, activation_fn, lamda):
        '''
        Compute and return gradients loss with respect to weights and biases.
        (dL/dW and dL/db)
        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
            y : Output of the network, numpy array of shape m x num_classes
            lamda : Regularization parameter.
        Returns
        ----------
            del_W : derivative of loss w.r.t. all weight values (a list of matrices).
            del_b : derivative of loss w.r.t. all bias values (a list of vectors).
        Hint: You need to do a forward pass before performing backward pass.
        '''
        del_W = []
        del_b = []
        
        d_w_layer = self.a_states[-1] - y
#         d_w_layer = (-y/self.a_states[-1]+(1-y)/(1-self.a_states[-1]))*stable_softmax(self.h_states[-1],derivative=True)
    
        del_W_n = 1./batch_size*np.dot(self.a_states[-2].T,d_w_layer) + 2*lamda*self.weights[-1]
        del_b_n = 1./batch_size*np.sum(np.dot(self.a_states[-2].T,d_w_layer), axis=0).reshape(-1,1) + 2*lamda*self.biases[-1]
        
        d_b_layer = del_b_n
        
        del_W.insert(0, del_W_n)
        del_b.insert(0, del_b_n)
        
        for i in range(self.num_layers, 0, -1):
            # with respect to inactivated neuron
            # d_w_layer = np.multiply(np.dot(d_w_layer,self.weights[i].T), activation_fn(self.h_states[i], derivative=True))
            # d_b_layer = np.multiply(np.dot(d_b_layer,self.weights[i].T), activation_fn(self.h_states[i], derivative=True))
            # d_b_layer = d_w_layer

            # with respect to activated neuron
            if i==self.num_layers:
                # d_w_layer = np.multiply(np.dot(np.multiply(d_w_layer,stable_softmax(self.h_states[i+1], derivative=True)),self.weights[i].T), activation_fn(self.h_states[i], derivative=True))
                # d_b_layer = np.multiply(np.dot(np.multiply(d_b_layer.T,stable_softmax(self.h_states[i+1], derivative=True)),self.weights[i].T), activation_fn(self.h_states[i], derivative=True))
                d_w_layer = np.multiply(np.dot(d_w_layer,self.weights[i].T), activation_fn(self.h_states[i], derivative=True))
                d_b_layer = np.multiply(np.dot(d_b_layer.T,self.weights[i].T), activation_fn(self.h_states[i], derivative=True))
            else:
                d_w_layer = np.multiply(np.dot(np.multiply(d_w_layer,activation_fn(self.h_states[i+1], derivative=True)),self.weights[i].T), activation_fn(self.h_states[i], derivative=True))
                d_b_layer = np.multiply(np.dot(np.multiply(d_b_layer,activation_fn(self.h_states[i+1], derivative=True)),self.weights[i].T), activation_fn(self.h_states[i], derivative=True))    
            
            del_W_i = np.dot(self.a_states[i-1].T,d_w_layer) + 2*lamda*self.weights[i-1]
            del_b_i = np.sum(d_b_layer, axis=0).reshape(len(self.biases[i-1]),1) + 2*lamda*self.biases[i-1]

            del_W.insert(0,del_W_i)
            del_b.insert(0,del_b_i)
        
        return del_W, del_b
#         raise NotImplementedError

class Optimizer(object):
    '''
    '''

    def __init__(self, learning_rate, optimization, beta=0.9, gamma=0.999):
        '''
        Create a Gradient Descent based optimizer with given
        learning rate.
        Other parameters can also be passed to create different types of
        optimizers.
        Hint: You can use the class members to track various states of the
        optimizer.
        '''
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.epsilon = 1e-6
        self.t = 0
        self.w_states = []
        self.b_states = []
        self.w_momentum = []
        self.b_momentum = []
        self.optimization_algorithm = optimization
#         raise NotImplementedError

    def step(self, weights, biases, delta_weights, delta_biases):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
        '''
        
        if self.optimization_algorithm == 'SGD':      
            self.t+=1
            for i in range(len(weights)):
                weights[i] = weights[i] - self.learning_rate*delta_weights[i]
                biases[i] = biases[i] - self.learning_rate*delta_biases[i]
        
        if self.optimization_algorithm == 'RMSProp':
            if self.t == 0:
                for i in range(len(weights)):
                    a,b = weights[i].shape
                    self.w_states.append(np.zeros([a,b]))
                    a,b = biases[i].shape
                    self.b_states.append(np.zeros([a,b]))
            
            self.t+=1
            for i in range(len(weights)):
                self.w_states[i] = self.gamma*self.w_states[i] + (1-self.gamma)*np.multiply(delta_weights[i],delta_weights[i])
                weights[i] = weights[i] - (self.learning_rate/(np.sqrt(self.w_states[i])+self.epsilon))*delta_weights[i]
                self.b_states[i] = self.gamma*self.b_states[i] + (1-self.gamma)*np.multiply(delta_biases[i],delta_biases[i])
                biases[i] = biases[i] - (self.learning_rate/(np.sqrt(self.b_states[i])+self.epsilon))*delta_biases[i]

        
        if self.optimization_algorithm == 'Adam':
            if self.t == 0:
                for i in range(len(weights)):
                    a,b = weights[i].shape
                    self.w_states.append(np.zeros([a,b]))
                    self.w_momentum.append(np.zeros([a,b]))
                    a,b = biases[i].shape
                    self.b_states.append(np.zeros([a,b]))
                    self.b_momentum.append(np.zeros([a,b]))
            
            self.t+=1
            for i in range(len(weights)):
                self.w_momentum[i] = self.beta*self.w_momentum[i] + (1-self.beta)*delta_weights[i]
                self.w_states[i] = self.gamma*self.w_states[i] + (1-self.gamma)*np.multiply(delta_weights[i],delta_weights[i])
                weights[i] = weights[i] - (self.learning_rate/(np.sqrt(self.w_states[i]/(1-self.gamma**self.t))+self.epsilon))*(self.w_momentum[i]/(1-self.beta**self.t))
                self.b_momentum[i] = self.beta*self.b_momentum[i] + (1-self.beta)*delta_biases[i]
                self.b_states[i] = self.gamma*self.b_states[i] + (1-self.gamma)*np.multiply(delta_biases[i],delta_biases[i])
                biases[i] = biases[i] - (self.learning_rate/(np.sqrt(self.b_states[i]/(1-self.gamma**self.t))+self.epsilon))*(self.b_momentum[i]/(1-self.beta**self.t))

        return weights, biases
#         raise NotImplementedError

def cross_entropy_loss(y, y_hat):
    '''
    Compute cross entropy loss
    Parameters
    ----------
        y : targets, numpy array of shape m x 4
        y_hat : predictions, numpy array of shape m x 4
    Returns
    ----------
        cross entropy loss
    '''
    return -np.sum(np.multiply(y,np.log(y_hat)))
#     raise NotImplementedError

def get_one_hot_encoding_of_label(label, one_hot_encoding):
    encoding = np.zeros(len(one_hot_encoding))
    encoding[one_hot_encoding[label]]=1
    return encoding

def train(
    net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target, one_hot_encoding, activation_fn
):
    '''
    In this function, you will perform following steps:
        1. Run gradient descent algorithm for `max_epochs` epochs.
        2. For each bach of the training data
            1.1 Compute gradients
            1.2 Update weights and biases using step() of optimizer.
        3. Compute RMSE on dev data after running `max_epochs` epochs.
    Here we have added the code to loop over batches and perform backward pass
    for each batch in the loop.
    For this code also, you are free to heavily modify it.
    '''

    m = train_input.shape[0]
    dev_target = np.array([get_one_hot_encoding_of_label(x,one_hot_encoding) for x in dev_target.to_numpy()])
    for e in range(max_epochs):
        epoch_loss = 0.
        number_of_batches = 0
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = np.array([get_one_hot_encoding_of_label(x,one_hot_encoding) for x in train_target[i:i+batch_size].to_numpy()])
            pred = net(batch_input, activation_fn)

            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, batch_size, activation_fn, lamda)

            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = cross_entropy_loss(batch_target, pred)
            epoch_loss += batch_loss
            number_of_batches += 1

        epoch_losses.append(epoch_loss)
        dev_loss=cross_entropy_loss(dev_target, net(dev_input, activation_fn))
        dev_losses.append(dev_loss)
        print('Epoch: {}, Dev Loss: {}, Epoch Loss: {}'.format(e, dev_loss, epoch_loss))

        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        #       stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    
    dev_pred = net(dev_input, activation_fn)
    dev_ce_loss = cross_entropy_loss(dev_target, dev_pred)

    print('C.E. on dev data: {:.5f}'.format(dev_ce_loss))

def get_test_data_predictions(net, inputs, activation_fn):
    '''
    Perform forward pass on test data and get the final predictions that can
    be submitted on Kaggle.
    Write the final predictions to the part2.csv file.
    Parameters
    ----------
        net : trained neural network
        inputs : test input, numpy array of shape m x d
    Returns
    ----------
        predictions (optional): Predictions obtained from forward pass
                                on test data, numpy array of shape m x 1
    '''
    return net(inputs, activation_fn)
#     raise NotImplementedError

def read_classification_data():
    '''
    Read the train, dev, and test datasets
    '''
    
    train_input = pd.read_csv("/Users/manasgabani/Downloads/IITB/CS725/assignment/cs-725-autumn-2022-assignment-classification/train.csv")
    # train_input = pd.read_csv("https://raw.githubusercontent.com/sahasrarjn/cs725-2022-assignment/main/regression/data/train.csv")
    train_target = train_input.iloc[:,0]
    train_input.drop(columns=train_input.columns[0],
                 axis=0,
                 inplace=True)
    
    dev_input = pd.read_csv("/Users/manasgabani/Downloads/IITB/CS725/assignment/cs-725-autumn-2022-assignment-classification/dev.csv")
    # dev_input = pd.read_csv("https://raw.githubusercontent.com/sahasrarjn/cs725-2022-assignment/main/regression/data/dev.csv")
    dev_target = dev_input.iloc[:,0]
    dev_input.drop(columns=dev_input.columns[0],
                 axis=0,
                 inplace=True)
    
    test_input = pd.read_csv("/Users/manasgabani/Downloads/IITB/CS725/assignment/cs-725-autumn-2022-assignment-classification/test.csv")
    # test_input = pd.read_csv("https://raw.githubusercontent.com/sahasrarjn/cs725-2022-assignment/main/regression/data/test.csv")

    return train_input, train_target, dev_input, dev_target, test_input

def feature_scaling(input_df, method='min_max_normalization', rescaling_range = (-1,1)):
    for col in input_df:
        mean = input_df[col].mean()
        std = input_df[col].std()
        min_value = input_df[col].min()
        max_value = input_df[col].max()
        if method=='min_max_normalization':
            input_df[col] = (input_df[col]-min_value)/(max_value - min_value)
        elif method=='min_max_normalization_with_rescaling':
            input_df[col] = rescaling_range[0] + ((input_df[col]-min_value)*(rescaling_range[1]-rescaling_range[0])/(max_value - min_value))
        elif method=='mean_normalization':
            input_df[col] = (input_df[col]-mean)/(max_value - min_value)
        elif method=='z_score_normalization':
            input_df[col] = (input_df[col]-mean)/std
        else:
            raise Exception('no such normalization method implemented')
    return input_df

def main():
    print('--- nn_classification.py ---')
    # Hyper-parameters 
    max_epochs = 75
    batch_size = 32
    # learning_rate = 0.001
    learning_rate = 0.007
    num_layers = 1
    num_units = [64]
    lamda = 0.01 # Regularization Parameter

    df = pd.DataFrame({
        'Name': ['learning_rate', 'batch_size', 'num_layers', 'num_units', 'regularization_parameter'],
        'Value': [learning_rate, batch_size, num_layers, num_units, lamda]
    })
    
    df.to_csv('./params.csv', index=False)

    classification_train_input, classification_train_target, classification_dev_input, classification_dev_target, classification_test_input = read_classification_data()
    
    classification_train_input = feature_scaling(classification_train_input, method='z_score_normalization')
    classification_dev_input = feature_scaling(classification_dev_input, method='z_score_normalization')
    classification_test_input = feature_scaling(classification_test_input, method='z_score_normalization')

    classes = set(classification_train_target)
    num_classes = len(classes)
    one_hot_encoding = {}
    inverse_one_hot = {}
    for i,label in enumerate(classes):
        one_hot_encoding[label] = i
        inverse_one_hot[i] = label

    net = Net(num_layers, num_units, num_classes)
    
    optimizer = Optimizer(learning_rate=learning_rate, optimization='SGD')
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        classification_train_input, classification_train_target,
        classification_dev_input, classification_dev_target, one_hot_encoding, relu
    )
#     get_test_data_predictions(net, test_input)
    test_pred = net(classification_test_input, relu)
    indices = test_pred.argmax(axis=1)
    test_label_pred = []
    for i, row in enumerate(test_pred.argmax(axis=1)):
        test_label_pred.append(inverse_one_hot[indices[i]])

    df = pd.DataFrame({
        'Id': list(classification_test_input.index+1),
        'Predictions': test_label_pred
    })

    df.to_csv('./22M0781.csv',index=False)

if __name__ == '__main__':
    main()