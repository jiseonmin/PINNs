# writing PINN that uses Tensorflow 2 based on Shota Deguchi's github
# Getting this error:
# ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.

import os
import time
import datetime
import tensorflow as tf
import numpy as np

class PhysicsInformedNN:
    def __init__(self, f, t, p, s0, N0, Rm, Rn, Rl, depth):

        X = np.concatenate([f, t], 1)
        # lower bound (lb) and upper bound (ub) are used to standardize the input values
        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X = X

        self.Rm = Rm # input dimension
        self.Rn = Rn # output dimension
        self.Rl = Rl # internal dimension
        self.depth = depth # number of hidden layers + output layer

        self.f = X[:, 0:1]
        self.t = X[:, 1:2]

        self.p = p

        self.s = tf.Variable(s0, dtype = tf.float32)
        self.N = tf.Variable(N0, dtype = tf.float32)

        self.dnn = self.dnn_init(Rm, Rn, Rl, depth)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    def random_seed(self, seed = 1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def dnn_init(self, Rm, Rn, Rl, depth):
        network = tf.keras.Sequential()
        network.add(tf.keras.layers.InputLayer(Rm))
        # Standardize the input points
        network.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))
        w_init = tf.Variable(tf.random.truncated_normal([Rl, Rl], stddev=np.sqrt(1/Rl), dtype=tf.float32))
        b_init = tf.Variable(tf.zeros([1,Rl], dtype=tf.float32), dtype=tf.float32)
        for l in range(depth - 1):
            network.add(tf.keras.layers.Dense(Rl, activation = tf.keras.activations.get('tanh'), use_bias = False, 
            kernel_initializer = 'glorot_normal', 
            kernel_regularizer = None, bias_regularizer = None,
            activity_regularizer = None, kernel_constraint = None, bias_constraint = None))

        
        network.add(tf.keras.layers.Dense(Rn, activation = tf.keras.activations.softplus))
        return network

    def net_NS(self, f, t):
        t = tf.convert_to_tensor(t, dtype = tf.float32)
        f = tf.convert_to_tensor(f, dtype = tf.float32)
        s = self.s
        N = self.N
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(t)
            tp.watch(f)

            p = self.dnn(tf.concat([f, t], 1))
        p_f = tp.gradient(p, f)
        p_t = tp.gradient(p, t)
        p_ff = tp.gradient(p_f, f)
        del tp
        g = p_t + s * ((1-2*f)*p + (f-f**2)*p_f) + 1/(2*N)*(-2*p + 2*(1-2*f)*p_f + (f-f**2)*p_ff)
        return p, g

    def compute_loss(self, f_data, t_data, p_data):
        p_pred, g_pred = self.net_NS(f_data, t_data)
        loss = tf.reduce_mean(tf.square(p_data - p_pred)) + tf.reduce_sum(tf.square(g_pred))
        return loss
    

    def loss_grad(self, f_data, t_data, p_data):
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(self.s)
            tp.watch(self.N)
            loss = self.compute_loss(f_data, t_data, p_data)
        grad = tp.gradient(loss, self.s, self.N)
        del tp
        return loss, grad

    def grad_desc(self, f_data, t_data, p_data):
        loss, grad = self.loss_grad(f_data, t_data, p_data)
        self.optimizer.apply_gradients(zip(grad, self.s, self.N))
        return loss

    def train(self, epoch = 10 ** 5):
        print(">>>> training setting;")
        print("     # of epoch   :", epoch)

        t0 = time.time()

        f_data = self.f
        t_data = self.t
        p_data = self.p
        for ep in range(epoch):
            ep_loss = self.grad_desc(f_data, t_data, p_data)
            if ep % 10 == 0:
                elapsed = time.time() - t0
                s_value = self.s
                N_value = self.N
                print('It: %d, Loss : %.3e, s: %.5f, N: %.1f, Time: %.2f' %
                (ep, ep_loss, s_value, N_value, elapsed))
        print("training ended, end time ", datetime.datetime.now())
    def predict(self, f, t):
        p_hat, g_hat = self.net_NS(t, f)
        return p_hat, g_hat
    

                
        

if __name__ == "__main__": 
      
    N_train = 5000
    Rm = 2
    Rl = 20
    Rn = 2
    depth = 4
    s0 = 0.01
    N0 = 500

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
    model = PhysicsInformedNN(f_train, t_train, phi_train, s0, N0, Rm, Rn, Rl, depth)
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
    
       
