# 


# To make it work, change the variable path line ~340, and install module if necessary 

## Import

# install numpy scipy pyqt matplotlib pandas sympy opencv
import numpy as np
import matplotlib.pyplot as plt
import time 
import random as rd
import cv2

## Required functions

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

## Core functions

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) 
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
    
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
    
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation='relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation='sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (-1./ m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log( 1-AL)))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost
    
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, cache[0].T) 
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(cache[1].T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
    
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
    
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

## Path !! YOU HAVE TO CHANGE IT DO YOUR CONFIGURATION !!

path="C:/Users/dimit/Documents/GitHub/Code_perso/Code_stage/stage_saillance_fusion_neuro" # Use : '/', Don't use : '\' 

## Creating eye_tracker_simplified in 0 and 1

def create_eye_tracker_simplified():
    for k in range(1,7+1): #Number of video
        for i in range (1, 30+1): #Number of picture in each video, limited by eye_tracker
            img = cv2.imread(path+"/v"+str(k)+"/eye_tracker/x"+str(i)+".jpg", 0) # 0 means gray_scale
            
            cv2.imwrite(path+"/v"+str(k)+"/eye_tracker/xs"+str(i)+".jpg",simplify_img(img,2,1))
            
            cv2.imwrite(path+"/v"+str(k)+"/eye_tracker/xs_visual_"+str(i)+".jpg",simplify_img(img,2,255))



def simplify_img(img, seuil_division, high): # high = 1 or 255
    # First, we find the maximum pixel in the picture
    temp_max = img[0][0]
    for i in range (0, img.shape[0], 5): #i+=5 pour aller plus vite
        for j in range(0, img.shape[1], 5): #j+=5 pour aller plus vite
            if img[i][j] > temp_max :
                temp_max = img[i][j]
    max = temp_max
    
    #Then, we change pixel in either 0 or 1/255
    for i in range (0, img.shape[0]): 
        for j in range(0, img.shape[1]): 
            if img[i][j] > (max/seuil_division):
                img[i][j] = high # Put 255 if you want to cv2.imshow('image',img) and see what it is doing
            else:
                img[i][j] = 0
    return img

        
## Create the library "images" with all image
# Launch this code one time to create the librairy images
def create_librairy_images():
    images={}
    for k in range(1,7+1): #Number of video
        for i in range (1, 30+1): #Number of picture in each video, limited by eye_tracker
            
            img_i =  cv2.imread(path+"/v"+str(k)+"/intensity/i"+str(i)+".jpg", 0) 
            images["v"+str(k)+"i"+str(i)] = img_i
            
            img_c =  cv2.imread(path+"/v"+str(k)+"/color/c"+str(i)+".jpg", 0) 
            images["v"+str(k)+"c"+str(i)] = img_c
            
            img_m =  cv2.imread(path+"/v"+str(k)+"/motion/m"+str(i)+".jpg", 0) 
            images["v"+str(k)+"m"+str(i)] = img_m
            
            img_o =  cv2.imread(path+"/v"+str(k)+"/orientation/o"+str(i)+".jpg", 0) 
            images["v"+str(k)+"o"+str(i)] = img_o
            
            img_xs = cv2.imread(path+"/v"+str(k)+"/eye_tracker/xs"+str(i)+".jpg", 0) 
            images["v"+str(k)+"xs"+str(i)] = img_xs
        

## Create librairy x and y

def create_x_and_y(images, steps ,shuffle):
    
    x,y={},{}
    
    #First, we create x and y librairy thanks to index and a group of pixels
    compt=0
    index=[]
    for k in range(1,1+7): #(1,7+1) Number of video
        for l in range (1, 30+1): #(1, 30+1) Number of picture in each video, limited by eye_tracker
    
            for i in range (0, images["v1i1"].shape[0],steps): # lign
                for j in range(0, images["v1i1"].shape[1],steps): # colomn
                    pixel_i = images["v"+str(k)+"i"+str(l)][i][j]
                    pixel_c = images["v"+str(k)+"c"+str(l)][i][j]
                    pixel_m = images["v"+str(k)+"m"+str(l)][i][j]
                    pixel_o = images["v"+str(k)+"o"+str(l)][i][j]
                    pixel_xs = images["v"+str(k)+"xs"+str(l)][i][j]
                    
                    array_x=np.array([[i,j,pixel_i,pixel_c,pixel_m,pixel_o]]) #two [[]] against weird "rank one" array with numpy librairy
                    array_y=np.array([[pixel_xs]])
                    
                    x[str(compt)]=array_x
                    y[str(compt)]=array_y
                    
                    index.append(compt)
                    compt+=1
                    
    #Then, we shuffle the index
    if shuffle:
        rd.shuffle(index)
        
        x_shuffle,y_shuffle={},{}
        compt=0
        for k in range(1,7+1): #(1,7+1) Number of video
            for l in range (1, 30+1): #(1, 30+1) Number of picture in each video, limited by eye_tracker
                for i in range (0, images["v1i1"].shape[0],steps): # lign
                    for j in range(0, images["v1i1"].shape[1],steps): # colomn
                        x_shuffle[str(index[compt])] = x[str(compt)]
                        y_shuffle[str(index[compt])] = y[str(compt)]
                        compt+=1
                        
        assert( len(x)==len(y)==len(x_shuffle)==len(y_shuffle) )
        return x_shuffle,y_shuffle
        
    else:
        assert(len(x)==len(y))
        return x,y
    
    
    
x,y=create_x_and_y(images, 300, True)

## Create x_train, y_train, x_test, y_test

def create_train_test(x,y,pourcentage_of_test=0.8): #pourcentage_of_test

    assert(len(x)==len(y))
    
    train_size=int(len(x)*pourcentage_of_test)
    test_size=len(x) - train_size
    print("Train size : "+str(train_size)+", and test size : " + str(test_size) )
    
    x_train, y_train = x[str(1)].T, y[str(1)].T
    x_test, y_test = x[str(train_size+1)].T, y[str(train_size+1)].T
    
    for k in range(1,train_size):
        x_train = np.concatenate((x_train, x[str(k)].T), axis=1)
        y_train = np.concatenate((y_train, y[str(k)].T), axis=1)
        
    for k in range(train_size+2,len(x)):
        x_test = np.concatenate((x_test, x[str(k)].T), axis=1)
        y_test = np.concatenate((y_test, y[str(k)].T), axis=1)
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = create_train_test(x,y,pourcentage_of_test=0.8)
print(x_train.shape)