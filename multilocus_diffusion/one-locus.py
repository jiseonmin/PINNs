
## Based on Navier-Stokes equation example, I want to test if I can make a PINN solve a single-locus PDE with selection and drift (fixsed population size)
## The equation goes as phi(f,t)_t = - d/df (sf(1-f)phi) + 1/2N d^2/df^2(f(1-f)phi)


import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
import pickle
np.random.seed(1234)
tf.disable_v2_behavior()
#tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, f, t, phi, layers):
        
        X = np.concatenate([f, t], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.f = X[:,0:1]
        self.t = X[:,1:2]
        
        self.phi = phi
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize parameters
        self.s = tf.Variable([0.0], dtype=tf.float32)
        self.N = tf.Variable([0.0], dtype=tf.float32)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.f_tf = tf.placeholder(tf.float32, shape=[None, self.f.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.phi_tf = tf.placeholder(tf.float32, shape=[None, self.phi.shape[1]])
        
        self.phi_pred, self.g_pred = self.net_NS(self.f_tf, self.t_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.phi_tf - self.phi_pred)) + \
                    tf.reduce_sum(tf.square(self.g_pred))
                    
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
        #                                                         method = 'L-BFGS-B', 
        #                                                         options = {'maxiter': 50000,
        #                                                                    'maxfun': 50000,
        #                                                                    'maxcor': 50,
        #                                                                    'maxls': 50,
        #                                                                    'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-5)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, f, t):
        s = self.s
        N = self.N
        
        phi = self.neural_net(tf.concat([f,t], 1), self.weights, self.biases)

        phi_f = tf.gradients(phi, f)[0]
        phi_t = tf.gradients(phi, t)[0]
        phi_ff = tf.gradients(phi_f, f)[0]

        g = phi_t + s * ((1-2*f)*phi + (f-f**2)*phi_f) + 1/(2*N)*(-2*phi + 2*(1-2*f)*phi_f + (f-f**2)*phi_ff)        
   
        return phi, g
    
    def callback(self, loss, s, N):
        print('Loss: %.3e, s: %.5f, N: %.2f' % (loss, s, N))
      
    def train(self, nIter): 

        tf_dict = {self.f_tf: self.f, self.t_tf: self.t, self.phi_tf: self.phi}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                s_value = self.sess.run(self.s)
                N_value = self.sess.run(self.N)
                print('It: %d, Loss: %.3e, s: %.5f, N: %.1f, Time: %.2f' % 
                      (it, loss_value, s_value, N_value, elapsed))
                start_time = time.time()
            
        self.optimizer_Adam.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.s, self.N],
                                loss_callback = self.callback)
            
    
    def predict(self, f_star, t_star):
        
        tf_dict = {self.f_tf: f_star, self.t_tf: t_star}
        
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        
        return phi_star

####################################################################################################
######## Re-write the plotting part (plot phi(f, t) with f(t=0) = 0.5, s = some constant))###########
####################################################################################################
        
if __name__ == "__main__": 
      
    N_train = 5000
    
    layers = [2, 20, 2]
    
    # Load Data # replace with something else (also will make .pkl file)
    data = np.load('single_locus_data.npy', allow_pickle='TRUE')
    # with open("single_locus_data.pkl", "wb") as tf:
    #     data = pickle.load(tf)    
    f = data.item()['f'][:, None]
    t = data.item()['t'][:, None]
    phi = data.item()['phi'][:, None]
    print(len(f))
    idx_list = np.arange(len(f))
    np.random.shuffle(idx_list)
    print(idx_list)

    f_train = f[idx_list[:N_train]]
    t_train = t[idx_list[:N_train]]
    phi_train = phi[idx_list[:N_train]]
    f_test = f[idx_list[N_train:]]
    t_test = t[idx_list[N_train:]]
    phi_test = phi[idx_list[N_train:]]

    # Training
    model = PhysicsInformedNN(f_train, t_train, phi_train, layers)
    model.train(200000)

    # Prediction  
    phi_pred = model.predict(f_test, t_test)
    s_value = model.sess.run(model.s)
    N_value = model.sess.run(model.N)
    
    # Error
    error_phi = np.linalg.norm(phi_train-phi_pred,2)/np.linalg.norm(phi_train,2)
    
    s_true = 0.001
    N_true = 1000
    error_s = np.abs(s_value - s_true)/s_true*100
    error_N = np.abs(N_value - N_true)/N_true * 100
    
    print('Error phi: %e' % (error_phi))    
    print('Error s: %.5f%%' % (error_s))                             
    print('Error N: %.5f%%' % (error_N))                  
    
