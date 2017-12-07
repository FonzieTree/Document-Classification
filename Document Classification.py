# problem from https://www.hackerrank.com/challenges/document-classification/problem
# data can be downloaded from https://s3.amazonaws.com/hr-testcases/597/assets/trainingdata.txt
# December 2017 by Shuming Fang. 
# fangshuming519@gmail.com.
# https://github.com/FonzieTree
import numpy as np
np.random.seed(1)
nb_of_samples = 5485
seq_len = 30
categories = 8
X = np.zeros((nb_of_samples,seq_len))
Y = np.zeros((nb_of_samples,categories))
with open("trainingdata.txt") as f:
    i = 0
    for line in f:
        num = int(line[0])
        Y[i,num-1] = 1
        seq = [ord(i) for i in line.strip()[2:]]
        index = min(seq_len,len(seq))
        X[i][:index] = seq[:index]
        i = i + 1
#####################################
X = X/122.0
# Building deep neural network
y = np.argmax(Y,axis=1)
N = nb_of_samples# number of points per class
D = seq_len # dimensionality
K = categories # number of classes
# initialize parameters randomly
h = 10 # size of hidden layer
w1 = 0.01 * np.random.randn(D,h)
b1 = np.zeros((1,h))
w2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
    # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1) # note, ReLU activation
    scores = np.dot(hidden_layer, w2) + b2
  
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(w1*w1) + 0.5*reg*np.sum(w2*w2)
    loss = data_loss + reg_loss
    
    if i % 10 == 0:
        acc = 100*sum(np.argmax(probs,axis = 1) == y)/N
        print("iteration %d: loss %f acc: %f" % (i, loss, acc))
    
  
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
  
    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dw2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, w2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dw1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)
  
    # add regularization gradient contribution
    dw2 += reg * w2
    dw1 += reg * w1
  
    # perform a parameter update
    w1 += -step_size * dw1
    b1 += -step_size * db1
    w2 += -step_size * dw2
    b2 += -step_size * db2
# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, w1) + b1)
scores = np.dot(hidden_layer, w2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
