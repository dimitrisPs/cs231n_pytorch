import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for example in range(num_samples):  
      scores = X[example].dot(W)
      scores_exp =np.exp(scores)
      num = np.exp(scores[y[example]])
      den = np.sum(np.exp(scores))
      p =num/den
      loss += -np.log(p)
      #http://cs231n.github.io/neural-networks-case-study/#grad
      #https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
      #https://stackoverflow.com/questions/41663874/cs231n-how-to-calculate-gradient-for-softmax-loss-function
      for cls in range(num_classes):
        
            dW[:, cls] += X[example] * (scores_exp[cls]/den - (cls == y[example])) 
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_samples
  loss += reg * np.sum(W*W)
    
  dW /=  num_samples
  dW += 2* reg * W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  #hot-1 encoding y
  y_enc = np.zeros((num_samples, num_classes))
  y_enc[np.arange(num_samples), y] = 1
  scores = X.dot(W)

  scores_exp =np.exp(scores)
  num = np.sum(scores_exp * y_enc, axis =1)
  den = np.sum(scores_exp, axis = 1)
  p = num/den
  loss = np.sum(-np.log(p)) / num_samples +reg * np.sum(W*W)
    
  # grad
  # The transpose matrices are used for broadcasting. The formula is the same as
  # naive but vectorized.
  a = (np.exp(scores).T /den).T - y_enc
  dW = X.T.dot( a)
  dW /=  num_samples
  dW += 2* reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

