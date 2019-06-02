import numpy as np
from random import shuffle

def predict_svm(X, w):
    return X.dot(w)


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = predict_svm(X[i], W) #10x1
    correct_class_score = scores[y[i]] #the predicted value of the correct class
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # To calculate the gradient we start from the loss function.
        # Li = sum{max(0, w_j'*x_i - w_{y_i}* w_i +1)}s
        # Differentiating the above expression w.r.t. w_j and w_{y_i}, results to  
        # dw_j = += x_i
        # dw_{y_i} -= x_i
        # respectively. Because we are working with batches, we add the gradient 
        # for each example and at the end we devide with the number of examples.
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # since the regularization term is W^2, if we defferentiate we get 2 * W
  dW += 2 * reg * W
    

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
    
  # magic numbers
  num_examples = y.size
  num_classes = W.shape[1]
    
  # convert labels to hot-1 encoding.
  y_encode = np.zeros((num_examples, num_classes))
  y_encode[np.arange(num_examples), y] = 1

  # In vecotrized implementation we will not use a max function nor a if statement.
  # one solution to do this is to treat all elements the same. So delta for s_yi
  # elements would be 0 and 1 for everything else. delta would be the sumplement 
  # to the hot-1 matrix. 
  delta = np.ones(y_encode.shape) - y_encode

  s = predict_svm(X, W)
    
  # s_y is a matrix that for every example has a row filled with the prediction
  # of the correct class. 
  s_y = (np.ones(s.shape).T * s[np.arange(num_examples), y]).T
  
  # The way we defined delta, takes care of the j!=y_i in the sum
  loss_g = (s - s_y + delta)

  #The above line implements the max function.
  valid_mask = loss_g>0
  
  # Sum non zero values and devide with the number of examples.
  loss = np.sum(loss_g[valid_mask])/num_examples
  # Add the regularization term
  loss += reg * np.sum(W * W)
   
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # The analytic expression to compute the gradient is given by the formula
  # dL/w_j = { 1 if (s>0) else 0} * xi.  when j!=y_i
  # dl/w_{y_i} = -sum{ 1 if (s>0) else 0} * xi when j==y_i
  # we will use the expression  { 1 if (s>0) else 0} to compute the matrix w_d,
  # ignoring the different expression got j==y_i. We can do that using the 
  # valid_mask computed before
  w_d = np.ones(s.shape)*valid_mask
  
  # With the w_d computed, we then compute only the elements from this matrix
  # that j==y_i using the formula -sum{ 1 if (s>0) else 0}
  w_d[np.arange(num_examples), y] = -np.sum(valid_mask, axis =1)

  #Now we multiply with the last term of the above formula xi, but because we have
  # more than one example, we take compute X.T.dot(w_d) and we devide with number 
  # examples
  dW = X.T.dot(w_d)
  dW /= num_examples

  #we add the regularization term
  dW +=2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
