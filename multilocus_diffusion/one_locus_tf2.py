# writing PINN that uses Tensorflow 2 based on Shota Deguchi's github
# Input : series of (f, t) observed from Wright-Fisher simulation.
# Aim : solve FP equation with selection and drift (dp(f,t)/dt = - d/df (sf(1-f)p) + 1/2N d^2/df^2(f(1-f)p))
# Use KL divergence loss function + data loss function as Chen et al. (2020) did. 

### todo : write __init__, write main.py part

import os
import time
import datetime
import tensorflow as tf
import numpy as np

class PINN(tf.keras.Model):
    def __init__(self, t_0, x_0, u_0, 
                Rm, Rn, Rl, depth, inv == True, s=0.01, N = 1000)

    self.Rm = Rm # input dimension
    self.Rn = Rn # output dimension
    self.Rl = Rl # internal dimension
    self.depth = depth # number of hidden layers + output layer
    self.s = s
    self.N = N
    if self.inv == True:
        self.s = tf.Variable(self.s, dtype = self.data_type)
        self.N = tf.Variable(self.N, dtype = self.data_type)
        self.params.append(self.s)
        self.params.append(self.N)
        self.s_log = []
        self.N_log = []
    elif self.inv == False:
        self.s = tf.constant(self.s, dtype=self.data_type)
        self.N = tf.constant(self.N, dtype=self.data_type)

    def random_seed(self, seed = 1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def dnn_init(self, Rm, Rn, Rl, depth):
        network = tf.keras.Sequential()
        network.add(tf.keras.layers.InputLayer(Rm))
        # Standardize the input points
        network.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))

        if self.BN == TRUE: 
            for l in range(depth - 1):
                network.add(tf.keras.layers.Dense(Rl, activation = self.activ, use_bias = False, 
                kernel_initializer = self.w_init, bias_initializer = self.b_init, 
                kernel_regularizer = None, bias_regularizer = None,
                activity_regularizer = None, kernel_constraint = None, bias_constraint = None))

                network.add(tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001, 
                center = True, scale = True, beta_initializer = "zeros", gamma_initializer = "ones", 
                moving_mean_initializer = "zeros", moving_variance_initializer = "ones", 
                beta_regularizer = None, gamma_regularizer = None, beta_constraint = None, gamma_constraint = None))
        
        else:
            for l in range(depth - 1):
                network.add(tf.keras.layers.Dense(Rl, activation = self.activ, use_bias = True, 
                kernel_initializer = self.w_init, bias_initializer = self.b_init, 
                kernel_regularizer = None, bias_regularizer = None, 
                activity_regularizer = None, kernel_constraint = None, bias_constraint = None))
        
        network.add(tf.keras.layers.Dense(Rn, activation = tf.keras.activations.softplus))
        return network
    def opt_alg(self, lr, opt):
        if opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.0, nesterov = False)
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizer.RMSprop(learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False)
        elif opt == "Adam":
            optimizer = tf.keras.optimizer.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
        elif opt == "Adammax":
            optimizer = tf.keras.optimizer.Adamax(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        elif opt == "Nadam":
            optimizer = tf.keras.optimizer.Nadam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        else:
            raise Exception("optimizer not specified correctly")
        return optimizer

    def PDE(self, t, f):
        t = tf.convert_to_tensor(t, dtype = self.data_type)
        f = tf.convert_to_tensor(f, dtype = self.data_type)
        s = self.s
        N = self.N
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(t)
            tp.watch(f)

            p = self.dnn(tf.concat([t, f], 1))
        p_f = tp.gradient(p, f)
        p_t = tp.gradient(p, t)
        p_ff = tp.gradient(p_f, f)
        del tp
        g = p_t + s * ((1-2*f)*p + (f-f**2)*p_f) + 1/(2*N)*(-2*p + 2*(1-2*f)*p_f + (f-f**2)*p_ff)
        return p, g

    ### loss function is defined as L_all = tau * L_pde + L_data. tau is a weight to balance pde loss and data loss.
    ### data loss is defined as Eqn 14 in Chen et al. 

    def loss_pde(self, t, f):
        dummy, g_hat = self.PDE(t, f)
        loss_pde = tf.reduce_mean(tf.square(g_hat))
        return loss_pde

    def loss_data(self, t, f):
        p_hat, dummy = self.PDE(t, f)
        loss_data = -tf.reduce_mean(tf.math.log(p_hat)) + tf.reduce_sum((p_hat[:-1]+p_hat[1:]) / 2 * (f[1:]-f[:-1])) + p_hat[0] / 2 * f[0] + (1 - p_hat[-1]) /2 * f[-1]
        return loss_data

    def loss_all(self, t_r, f_r, t_data, f_data):
        loss_pde = self.w_r * self.loss_pde(t_r, f_r)
        loss_data = self.w_data * self.loss_data(t_data, f_data)
        loss_all = loss_pde + loss_data
        return loss_all

    def loss_grad(self, t_r, f_r, t_data, f_data):
        with tf.GradientTape(persistent = True) as tp:
            loss = self.loss_all(t_r, f_r, t_data, f_data)
        grad = tp.gradient(loss, self.params)
        del tp
        return loss, grad

    def grad_desc(self, t_r, f_r, t_data, f_data):
        loss, grad = self.loss_grad(t_r, f_r, t_data, f_data)
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss

    def train(self, epoch = 10 ** 5, batch = 2 ** 6, tol = 1e-5):
        print(">>>> training setting;")
        print("     # of epoch   :", epoch)
        print("     batch size   :", batch)
        print("     convergence tol:", tol)

        t0 = time.time()

        t_data = self.t_data.numpy()
        f_data = self.f_data.numpy()
        t_r = self.t_r.numpy()
        f_r = self.f_r.numpy()

        for ep in range(epoch):
            ep_loss = 0

            if batch == 0: # full-batch training
                ep_loss = self.grad_desc(t_data, f_data, t_r, f_r)
            else: # mini-batch training
                n_data = self.f_data.shape[0]
                idx_data = np.random.permutation(n_data)
                n_r = self.f_r.shape[0]
                idx_r = np.random.permutation(n_r)

                for idx in range(0, n_r, batch):
                    t_data_batch = tf.convert_to_tesor(t_data[idx_data[idx: idx + batch if idx + batch < n_data else n_data]], dtype = self.data_type)
                    f_data_batch = tf.convert_to_tesor(f_data[idx_data[idx: idx + batch if idx + batch < n_data else n_data]], dtype = self.data_type)
                    t_r_batch = tf.convert_to_tesor(t_r[idx_r[idx: idx + batch if idx + batch < n_r else n_r]], dtype = self.data_type)
                    f_r_batch = tf.convert_to_tesor(f_r[idx_r[idx: idx + batch if idx + batch < n_r else n_r]], dtype = self.data_type)
                    loss_batch = self.grad_desc(t_data_batch, f_data_batch, t_r_batch, f_r_batch)
                    ep_loss += loss_batch / int(n_r / batch)

                if ep % self.f_mntr == 0:
                    elps = time.time() - t0
                    ep_s = self.s.numpy()
                    ep_N = self.N.numpy()
                    self.ep_log.append(ep)
                    self.loss_log.append(ep_loss)
                    self.s_log.append(ep_s)
                    self.N_log.append(ep_N)
                    print("ep: %d, loss: %.3e, s: %.3f, N: %.1f" % (ep, ep_loss, ep_s, ep_N))
                    t0 = time.time()
        print("training ended, end time ", datetime.datetime.now())
    def predict(self, t, f):
        p_hat, g_hat = self.PDE(t, f)
        return p_hat, g_hat
    

                
        

        
