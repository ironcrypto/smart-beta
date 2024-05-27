import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import time
from datetime import datetime
from model.utils import utils
import config
from data import data
    
tfd = tfp.distributions

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()


class irl_market(object):
    def __init__(self, df_input, lr=0.0001, max_iter=5000, tol=1e-15,
                 n_actions=1, subgraph=True, saver=None, session=None, name='irl_market'):
        
        # mandatory DATA at init
        self.df_input = df_input
        
        # other
        self.tokens = data.get_calibPortfolioList()
        self.signals = config._SIGNALS
        self.ldate = config._LOWERDATE_HIST
        self.udate = config._UPPERDATE_HIST 
        self.timeVec = pd.date_range(start=self.ldate, end=self.udate, freq='D').to_frame()
        self.windows = config._WINDOWS
        self.n_actions = config._N_ACTIONS
        self.input_dims = (self.df_input.shape[0], len(self.tokens), len(self.signals))
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.name = name
        self.saver = saver
        self.sess = session
          
        if (session==None and subgraph==False):
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()
            
        self.chkpt_dir = ''.join([config.CPT_PATH,'/irl'])
        self.checkpoint_file = os.path.join (self.chkpt_dir,'irl_market.chkpt')
        

    def _create_placeholders(self):
        self.zprimet = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='z_signals')  # Tensor
        self.mu = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='mu') 
        self.wz = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='wz')
        self.WZ = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='WZ')
        self.scale = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='scale')
        self.WzScale = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='WzScale')
        self.Vt = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='Vt')
        self.loss = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='loss')
        self.xt_nrm=tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='xt_nrm')
        
        
    def _create_signals(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):

            
            #self.xt = self.df_input
            self.xt_avg = self.df_input.loc[self.ldate:self.udate, self.tokens].sum(axis=1).mean()  # scalar used to normalize
            self.xt_nrm = self.df_input.loc[self.ldate:self.udate, self.tokens] / self.df_input.mean()  ##DataFrame TxN

            tmp_list = []
            self.tmp_list_ = []
            const = False
            signal = False

            for s in range(self.input_dims[2]):
                # signals are built and normalized by the average over the focus period

                if self.signals[s] == 'const':
                    const = True

                else:
                    signal = True
                    if self.signals[s] == 'short_rolling' or self.signals[s] == 'long_rolling':
                        tmp_ = self.df_input.loc[self.ldate:self.udate, self.tokens].rolling(
                            window=self.windows[s]).mean()  # xt_nrm inst of xt
                        tmp_ = tmp_ / tmp_.loc[tmp_.first_valid_index()]  # normalize to the first MA (not NaN)
                        tmp_ = tmp_.pct_change(periods=1).shift(-1)  # take in PCT and reshift index to first valid

                    elif self.signals[s] == 'EWMA':
                        tmp_ = self.df_input.loc[self.ldate:self.udate, self.tokens].ewm(span=self.windows[s],
                                                                                   adjust=False).mean() / self.xt_avg
                        tmp_ = tmp_ / tmp_.loc[tmp_.first_valid_index()]  # normalize to the first MA (not NaN)
                        tmp_ = tmp_.pct_change(periods=1).shift(-1)  # take in PCT and reshift index to first valid

                    elif self.signals[s] == 'MACD' or self.signals[s] == 'MACD_signal':
                        ewma1 = self.df_input.loc[self.ldate:self.udate, self.tokens].ewm(span=6, adjust=False).mean()
                        ewma2 = self.df_input.loc[self.ldate:self.udate, self.tokens].ewm(span=12, adjust=False).mean()
                        macd = (ewma1 - ewma2)

                        if self.signals[s] == 'MACD':
                            tmp_ = macd.iloc[
                                   1:] / self.xt_avg  ### self.xt_avg #delete first row because it's zero and normalize

                        if self.signals[s] == 'MACD_signal':
                            tmp_ = macd.ewm(span=4, adjust=False).mean()
                            tmp_ = macd - tmp_
                            tmp_ = tmp_.iloc[1:] / self.xt_avg  # delete first row because it's zero and normalize

                    else:
                        print('Error - signal not yet supported - please create it in build_signals()')
                        break

                    tmp_list.append(tmp_.dropna(axis=0))

            idx = [f.index for f in tmp_list]  # select the indices of xt_nrm from the remaining indices in tmp_list
            for k in range(len(idx)):
                self.xt_nrm = self.xt_nrm[self.xt_nrm.index.isin(idx[k])]

            self.active_index = self.xt_nrm.index

            for k in range(len(tmp_list)):
                self.tmp_list_.append(tf.constant(tmp_list[k][tmp_list[k].index.isin(self.xt_nrm.index)], tf.float32))

            if const == True and signal == True:  # add a constant, only if 'const' and another signal are given
                tmp_ = tf.ones([*self.tmp_list_[0].shape], np.float32) / self.df_input.mean()  # self.xt_avg
                self.tmp_list_.insert(0, tmp_)

            self.zprimet = tf.stack(self.tmp_list_, axis=0)  # Tensor KxTxN
            #self.xt_nrm = tf.constant(self.xt_nrm, tf.float32)  # Tensor TxN

    def _create_trajectories(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):

            self.kappa = tf.compat.v1.get_variable("kappa", dtype=tf.float32,
                                               initializer=tf.compat.v1.random_uniform([self.input_dims[1]],
                                                                                       minval=0.0,
                                                                                       maxval=1.0))
            self.sigma = tf.compat.v1.get_variable("sigma", dtype=tf.float32,
                                               initializer=tf.compat.v1.random_uniform([self.input_dims[1]],
                                                                                       minval=0.0,
                                                                                       maxval=0.1))
            self.Wdiag = tf.compat.v1.get_variable("Wdiag", dtype=tf.float32,
                                               initializer=tf.compat.v1.random_normal([self.input_dims[1],
                                                                                       self.input_dims[2]],
                                                                                      mean=0.5,
                                                                                      stddev=0.1))
            self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


            self.mu = tf.zeros([self.input_dims[1]])  # vector of zero-Mean for the Multivariate dist
            xret = tf.divide(tf.subtract(tf.compat.v1.manip.roll(self.xt_nrm, shift=-1, axis=0),self.xt_nrm), self.xt_nrm)  # tensor TxN of relative returns - float64
            xret = tf.cast(xret, tf.float32)  # tensor TxN of relative returns - float32
            self.wz = tf.multiply(self.Wdiag, tf.transpose(self.zprimet, perm=[1, 2, 0]))  # NxK * TxNxK = TxNxK
            self.WZ = tf.reduce_sum(self.wz, axis=2)  # TxN #sum over the signals for each T and N
            self.scale = tf.slice(self.xt_nrm, [0, 0], [1, -1])  # 1xN # scale taken from first T slice
            self.WzScale = tf.multiply(self.scale, tf.compat.v1.cumprod(1 + self.WZ))  # TxN #rescale the signals and propagate geometric series from pct
            self.Vt = tf.subtract(xret, tf.multiply(self.kappa,tf.subtract(self.WzScale, self.xt_nrm)))  # TxN # portfolio value
            self.Vt = self.Vt[:-1, :]  # last item t=Tmax is discarded because no T+1 obs

    
    def _create_optimizer(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
        return self.train_op
    
    def _create_loss(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):         
            dist = tfd.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma)
            log_prob = dist.log_prob(self.Vt)
            reg_term = tf.reduce_sum(tf.square(self.Wdiag - 1))
            self.loss = -tf.reduce_sum(log_prob) + 0.01 * reg_term
        return self.loss
        
   
    def print_results(self):
        # Put data in pandas Dataframe.
        # Creating a Pandas Dataframe to hold the results
       
        rescol=['kappa','sigma','sigma^2',]
        for i in range(len(self.signals)):
            rescol.append(''.join(['W',str(i+1)]))
                                        
                                    
        results = pd.DataFrame( [],
                            index = self.tokens,
                            columns = rescol )
        results['kappa'] = self.sess.run(self.params[0])
        results['sigma'] = self.sess.run(self.params[1])
        results['sigma^2'] = self.sess.run(self.params[1])**2
        tmp=self.sess.run(self.params[2])
        for i in range(len(self.signals)):
            results[''.join(['W',str(i+1)])] = tmp[:,i]
        
        #save 
        results.to_csv(path_or_buf=os.path.join(self.chkpt_dir,'results.csv')) 
           
        print( "------------------- Calibration Results ----------------------" )
        print(results.round(4))
        
            

    
    def build_graph(self):
        #self._create_agent()
        self._create_placeholders()
        self._create_signals()
        self._create_trajectories()
        self._create_loss()
        self._create_optimizer()  
    
    class agent(object):
        def __init__(self, df_input, saver=None, session=None, name='agent'):
            
            self.saver = saver
            self.sess = session
            self.mem_cntr=0
            self.irl_mkt = irl_market(df_input = df_input,saver=self.saver, session=self.sess)
                                      #signals=self.signals, ldate=self.ldate, udate=self.udate,
                                      #tokens=self.tokens, windows=self.windows)
            
            
        def learn(self):
            irl_mkt_params, irl_mkt_loss, irl_mkt_train = self.irl_mkt.sess.run(
                [self.irl_mkt.params, self.irl_mkt.loss, self.irl_mkt.train_op])
            self.mem_cntr += 1
            return irl_mkt_params, irl_mkt_loss, irl_mkt_train

        # def save_models(self):
        #     self.irl_mkt.save_checkpoint()
        
        
        # def load_models(self):
        #     self.irl_mkt.load_checkpoint()        


        def run_steps(self,udate,load_checkpoint=False):
            
            with tf.compat.v1.variable_scope(self.irl_mkt.name, reuse=tf.compat.v1.AUTO_REUSE):
                self.irl_mkt.toc = time.process_time()
                
                self.irl_mkt.udate = udate
                
                #tf.compat.v1.reset_default_graph()
                tf.compat.v1.random.set_random_seed(42)
                
                self.irl_mkt.max_iter = 50000
                if load_checkpoint:
                    utils.load_checkpoint(self.irl_mkt.saver,self.irl_mkt.sess,self.irl_mkt.checkpoint_file)
                    
                losses = []
                losses.append(0)
                i = 1
            
                # Calibrate print out
                print("------------------- Calibration Calculating ----------------------")
                print(" iter |       Loss       |   difference")
            
                while True:
            
                    params, new_loss, b = self.learn()
                    loss_diff = np.abs(new_loss - losses[-1])
                    losses.append(new_loss)
            
                    if i % min(1000, (self.irl_mkt.max_iter / 20)) == 1:
                        print("{:5} | {:16.4f} | {:12.4f}".format(i, new_loss, loss_diff))
            
                    if loss_diff < self.irl_mkt.tol:
                        print('Loss function convergence in {} iterations!'.format(i))
                        print('Old loss: {}  New loss: {}'.format(losses[-2], losses[-1]))
                        utils.save_checkpoint(self.irl_mkt.saver,self.irl_mkt.sess,self.irl_mkt.checkpoint_file)
                        break
            
                    if i >= self.irl_mkt.max_iter:
                        print('Max number of iterations reached without convergence.')
                        utils.save_checkpoint(self.irl_mkt.saver,self.irl_mkt.sess,self.irl_mkt.checkpoint_file)
                        break
            
                    i += 1
                
                self.irl_mkt.print_results()  
    
                self.irl_mkt.tic = time.process_time()
                print ("dot = " + "\n ----- Computation time = " + str(1000*(self.irl_mkt.toc - self.irl_mkt.tic)) + "ms")              
                
            return self.irl_mkt.params, self.irl_mkt.WzScale, self.irl_mkt.active_index, self.irl_mkt.zprimet, self.irl_mkt.Wdiag
        
 
        

        
        
        
