# writing PINN that uses Tensorflow 2 based on Shota Deguchi's github
# Getting this error:
# ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.

import os
import time

from datetime import datetime
import tensorflow as tf
import numpy as np

####### custom LBFGS########

# Adapted from https://github.com/yaroslavvb/stuff/blob/master/eager_lbfgs/eager_lbfgs.py

import tensorflow as tf
import numpy as np
import time

# Time tracking functions
global_time_list = []
global_last_time = 0
def reset_time():
  global global_time_list, global_last_time
  global_time_list = []
  global_last_time = time.perf_counter()
  
def record_time():
  global global_last_time, global_time_list
  new_time = time.perf_counter()
  global_time_list.append(new_time - global_last_time)
  global_last_time = time.perf_counter()
  #print("step: %.2f"%(global_time_list[-1]*1000))

def last_time():
  """Returns last interval records in millis."""
  global global_last_time, global_time_list
  if global_time_list:
    return 1000 * global_time_list[-1]
  else:
    return 0

def dot(a, b):
  """Dot product function since TensorFlow doesn't have one."""
  return tf.reduce_sum(a*b)

def verbose_func(s):
  print(s)

final_loss = None
times = []
def lbfgs(opfunc, x, config, state, do_verbose, log_fn):
  """port of lbfgs.lua, using TensorFlow eager mode.
  """

  if config.maxIter == 0:
    return

  global final_loss, times
  
  maxIter = config.maxIter
  maxEval = config.maxEval or maxIter*1.25
  tolFun = config.tolFun or 1e-5
  tolX = config.tolX or 1e-19
  nCorrection = config.nCorrection or 100
  lineSearch = config.lineSearch
  lineSearchOpts = config.lineSearchOptions
  learningRate = config.learningRate or 1
  isverbose = config.verbose or False

  # verbose function
  if isverbose:
    verbose = verbose_func
  else:
    verbose = lambda x: None

    # evaluate initial f(x) and df/dx
  f, g = opfunc(x)

  f_hist = [f]
  currentFuncEval = 1
  state.funcEval = state.funcEval + 1
  p = g.shape[0]

  # check optimality of initial point
  tmp1 = tf.abs(g)
  if tf.reduce_sum(tmp1) <= tolFun:
    verbose("optimality condition below tolFun")
    return x, f_hist

  # optimize for a max of maxIter iterations
  nIter = 0
  times = []
  while nIter < maxIter:
    start_time = time.time()
    
    # keep track of nb of iterations
    nIter = nIter + 1
    state.nIter = state.nIter + 1

    ############################################################
    ## compute gradient descent direction
    ############################################################
    if state.nIter == 1:
      d = -g
      old_dirs = []
      old_stps = []
      Hdiag = 1
    else:
      # do lbfgs update (update memory)
      y = g - g_old
      s = d*t
      ys = dot(y, s)
      
      if ys > 1e-10:
        # updating memory
        if len(old_dirs) == nCorrection:
          # shift history by one (limited-memory)
          del old_dirs[0]
          del old_stps[0]

        # store new direction/step
        old_dirs.append(s)
        old_stps.append(y)

        # update scale of initial Hessian approximation
        Hdiag = ys/dot(y, y)

      # compute the approximate (L-BFGS) inverse Hessian 
      # multiplied by the gradient
      k = len(old_dirs)

      # need to be accessed element-by-element, so don't re-type tensor:
      ro = [0]*nCorrection
      for i in range(k):
        ro[i] = 1/dot(old_stps[i], old_dirs[i])
        

      # iteration in L-BFGS loop collapsed to use just one buffer
      # need to be accessed element-by-element, so don't re-type tensor:
      al = [0]*nCorrection

      q = -g
      for i in range(k-1, -1, -1):
        al[i] = dot(old_dirs[i], q) * ro[i]
        q = q - al[i]*old_stps[i]

      # multiply by initial Hessian
      r = q*Hdiag
      for i in range(k):
        be_i = dot(old_stps[i], r) * ro[i]
        r += (al[i]-be_i)*old_dirs[i]
        
      d = r
      # final direction is in r/d (same object)

    g_old = g
    f_old = f
    
    ############################################################
    ## compute step length
    ############################################################
    # directional derivative
    gtd = dot(g, d)

    # check that progress can be made along that direction
    if gtd > -tolX:
      verbose("Can not make progress along direction.")
      break

    # reset initial guess for step size
    if state.nIter == 1:
      tmp1 = tf.abs(g)
      t = min(1, 1/tf.reduce_sum(tmp1))
    else:
      t = learningRate


    # optional line search: user function
    lsFuncEval = 0
    if lineSearch and isinstance(lineSearch) == types.FunctionType:
      # perform line search, using user function
      f,g,x,t,lsFuncEval = lineSearch(opfunc,x,t,d,f,g,gtd,lineSearchOpts)
      f_hist.append(f)
    else:
      # no line search, simply move with fixed-step
      x += t*d
      
      if nIter != maxIter:
        # re-evaluate function only if not in last iteration
        # the reason we do this: in a stochastic setting,
        # no use to re-evaluate that function here
        f, g = opfunc(x)
        lsFuncEval = 1
        f_hist.append(f)


    # update func eval
    currentFuncEval = currentFuncEval + lsFuncEval
    state.funcEval = state.funcEval + lsFuncEval

    ############################################################
    ## check conditions
    ############################################################
    if nIter == maxIter:
      break

    if currentFuncEval >= maxEval:
      # max nb of function evals
      verbose('max nb of function evals')
      break

    tmp1 = tf.abs(g)
    if tf.reduce_sum(tmp1) <=tolFun:
      # check optimality
      verbose('optimality condition below tolFun')
      break
    
    tmp1 = tf.abs(d*t)
    if tf.reduce_sum(tmp1) <= tolX:
      # step size below tolX
      verbose('step size below tolX')
      break

    if tf.abs(f-f_old) < tolX:
      # function value changing less than tolX
      verbose('function value changing less than tolX'+str(tf.abs(f-f_old)))
      break

    if do_verbose:
      log_fn(nIter, f.numpy(), True)
      #print("Step %3d loss %6.5f msec %6.3f"%(nIter, f.numpy(), last_time()))
      record_time()
      times.append(last_time())

    if nIter == maxIter - 1:
      final_loss = f.numpy()


  # save state
  state.old_dirs = old_dirs
  state.old_stps = old_stps
  state.Hdiag = Hdiag
  state.g_old = g_old
  state.f_old = f_old
  state.t = t
  state.d = d

  return x, f_hist, currentFuncEval

# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)

##########################

class Logger(object):
  def __init__(self, frequency=10):
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

    self.start_time = time.time()
    self.frequency = frequency

  def __get_elapsed(self):
    return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

  def __get_error_u(self):
    return self.error_fn()

  def set_error_fn(self, error_fn):
    self.error_fn = error_fn
  
  def log_train_start(self, model):
    print("\nTraining started")
    print("================")
    self.model = model
    print(self.model.summary())

  def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
    if epoch % self.frequency == 0:
      print(f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  error = {self.__get_error_u():.4e}  " + custom)

  def log_train_opt(self, name):
    # print(f"tf_epoch =      0  elapsed = 00:00  loss = 2.7391e-01  error = 9.0843e-01")
    print(f"—— Starting {name} optimization ——")

  def log_train_end(self, epoch, custom=""):
    print("==================")
    print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()}  error = {self.__get_error_u():.4e}  " + custom)

class PhysicsInformedNN:
    def __init__(self, layers, optimizer, logger, ub, lb, s, N):

        # build the neural net
        # lower bound (lb) and upper bound (ub) are used to standardize the input values
        self.p_model = tf.keras.Sequential()
        self.p_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        self.p_model.add(tf.keras.layers.Lambda(
            lambda X: 2.0*(X - lb)/(ub - lb) - 1.0
        ))
        for width in layers[1:]:
            self.p_model.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh, kernel_initializer='glorot_normal'
            ))
        self.dtype = tf.float32

        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))
        self.s = tf.Variable([s], dtype=self.dtype)
        self.N = tf.Variable([N], dtype=self.dtype)


        self.optimizer = optimizer
        self.logger = logger

    

    ## model
    def __g_model(self, X):
        s, N = self.get_params()
        f_g = tf.convert_to_tensor(X[:, 0:1], dtype=self.dtype)
        t_g = tf.convert_to_tensor(X[:, 1:2], dtype=self.dtype)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(f_g)
            tape.watch(t_g)
            X_g = tf.stack([f_g[:, 0], t_g[:, 0]], axis=1)

            p = self.p_model(X_g)
            p_f = tape.gradient(p, f_g)
        p_ff = tape.gradient(p_f, f_g)
        p_t = tape.gradient(p, t_g)
        del tape
        s, N = self.get_params(numpy=True)
        g = p_t + s * ((1 - 2 * f_g) * p + (f_g - f_g ** 2) * p_f + 1 / (2 * N) * (-2 * p + 2 * (1 - 2 * f_g) * p_f + (f_g - f_g ** 2) * p_ff))
        return g

    def __loss(self, X, p, p_pred):
        g_pred = self.__g_model(X)
        return tf.reduce_mean(tf.square(p - p_pred)) + tf.reduce_mean(tf.square(g_pred))
    
    def __grad(self, X, p):
        with tf.GradientTape() as tape:
            loss_value = self.__loss(X, p, self.p_model(X))
        return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())
    
    def __wrap_training_variables(self):
        var = self.p_model.trainable_variables
        var.extend([self.s, self.N])
        return var


    def get_weights(self):
        w = []
        for layer in self.p_model.layers[1:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        w.extend(self.s.numpy())
        w.extend(self.N.numpy())
        return tf.convert_to_tensor(w, dtype=self.dtype)
    
    def set_weights(self, w):
        for i, layer in enumerate(self.p_model.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)
        self.s.assign([w[-2]])
        self.N.assign([w[-1]])
    
    def get_params(self, numpy=False):
        s = self.s
        N = self.N
        if numpy:
            return s.numpy()[0], N.numpy()[0]
        return s, N

    def summary(self):
        return self.p_model.summary()

    def fit(self, X, p, tf_epochs, nt_config):
        self.logger.log_train_start(self)
        X = tf.convert_to_tensor(X, dtype=self.dtype)
        p = tf.convert_to_tensor(p, dtype=self.dtype)

        def log_train_epoch(epoch, loss, is_iter):
            s, N = self.get_params(numpy=True)
            custom = f"s = {s:5f}, N = {N:2f}"
            self.logger.log_train_epoch(epoch, loss, custom, is_iter)
        
        self.logger.log_train_opt("Adam")
        for epoch in range(tf_epochs):
            loss_value, grads = self.__grad(X, p)
            self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
            log_train_epoch(epoch, loss_value, False)
        
        self.logger.log_train_opt("LBFGS")
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.set_weights(w)
                tape.watch(w)
                tape.watch(self.s)
                tape.watch(self.N)
                loss_value = self.__loss(X, p, self.p_model(X))
            grad = tape.gradient(loss_value, self.__wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat
        lbfgs(loss_and_flat_grad, self.get_weights(), nt_config, Struct(), True, log_train_epoch)
    
    def predict(self, X_star):
        p_star = self.p_model(X_star)
        g_star = self.__g_model(X_star)
        return p_star.numpy(), g_star.numpy()
    

                
        

if __name__ == "__main__": 
      
    N_train = 5000
    layers = [2, 20, 20, 20, 20, 2]
    s0 = 0.01
    N0 = 500
    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    logger = Logger(frequency=10)

    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    nt_epochs = 1000
    nt_config = Struct()
    nt_config.learningRate = 0.8
    nt_config.maxIter = nt_epochs
    nt_config.nCorrection = 50
    nt_config.tolFun = 1.0 * np.finfo(float).eps

    # Load Data # replace with something else (also will make .pkl file)
    data = np.load('single_locus_data.npy', allow_pickle='TRUE')
    # with open("single_locus_data.pkl", "wb") as tf:
    #     data = pickle.load(tf)    
    f = data.item()['f'][:, None]
    t = data.item()['t'][:, None]
    phi = data.item()['phi'][:, None]
    print(len(f))

    X = np.concatenate([f, t], 1)
        
    lb = X.min(0)
    ub = X.max(0)
                
    idx_list = np.arange(len(f))
    np.random.shuffle(idx_list)
    print(idx_list)

    f_train = f[idx_list[:N_train]]
    t_train = t[idx_list[:N_train]]
    X_train = np.concatenate([f_train, t_train], 1)
    p_train = phi[idx_list[:N_train]]
    f_test = f[idx_list[N_train:]]
    t_test = t[idx_list[N_train:]]
    X_test = np.concatenate([f_test, t_test], 1)
    p_test = phi[idx_list[N_train:]]


##layers, optimizer, logger, X_g, ub, lb, s, N
    # Training
    model = PhysicsInformedNN(layers, tf_optimizer, logger, ub, lb, s0, N0)
    tf_epochs = 100

    # Prediction  
    p_pred, g_pred = model.predict(X_test)

    s_pred, N_pred = model.get_params(numpy=True)
    
    # Error
    error_p = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)

    s_true = 0.001
    N_true = 1000
    error_s = np.abs(s_pred - s_true)/s_true*100
    error_N = np.abs(N_pred - N_true)/N_true * 100
    def error():
        return error_p
    logger.set_error_fn(error)    
    model.fit(X_train, p_train, tf_epochs, nt_config)

    error_p = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)

    error_s = np.abs(s_pred - s_true)/s_true*100
    error_N = np.abs(N_pred - N_true)/N_true * 100

    print('Error phi: %e' % (error_p))    
    print('Error s: %.5f%%' % (error_s))                             
    print('Error N: %.5f%%' % (error_N))                  
    
       
