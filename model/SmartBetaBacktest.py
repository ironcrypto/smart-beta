#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:29:52 2022

@author: stephanethomas
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
#from model.SmartBetaIndex import SmartBetaIndex
from model.irl_market import irl_market
import config
import utils.utils as utils



class SmartBetaBacktest(object):
    def __init__(self, SmartBeta, df_input, df_ohlc=None, saver=None, session=None, name='irl_bt', df_fitMeans=None, df_fitParams=None):
        
        # mandatory DATA at init
        self.df_input = df_input
        self.input_dims = (self.df_input.shape[0], len(SmartBeta.tokensCalib), len(SmartBeta.signals))
        
        
        # can be defined afterwards
        if df_fitMeans==None: 
            self.fitmeans=None
            self.fitparams=None
            self.initCalib=True
        else:
            self.fitmeans = df_fitMeans
            self.fitparams = df_fitParams
            self.initCalib=False
        
        self.openclose = df_ohlc
        self.name = name
        
        self.tokensIndex = SmartBeta.tokensIndex
        self.tokensCalib = SmartBeta.tokensCalib
        self.tokenN = len(SmartBeta.tokensIndex)
        self.contrib = SmartBeta.contrib
        self.rebalFreq = SmartBeta.rebalFreq
        self.rebalDay = SmartBeta.rebalDay
        self.recalibFreq = SmartBeta.recalibFreq
        self.signals = SmartBeta.signals
        self.ugscores = SmartBeta.ugscores
        
        self.initCalc = True
        
        self.ldate = config._LOWERDATE_HIST
        self.udate = config._UPPERDATE_HIST 
        self.ldateBT = config._STARTDATE_BT
        self.udateBT = config._ENDDATE_BT
        self.timeVec = pd.date_range(start=self.ldate, end=self.udateBT, freq='D').to_frame()
        self.timedf = self.timeVec[0][0]
        self.ind = 1
        self.windows = config._WINDOWS
        self.n_actions = config._N_ACTIONS
        self.includeTxCost=config._INCLUDE_TXCOST
        
        self.saver=saver
        self.sess=session
        
        
        
        ######
        self.dataIngestion() # hack until i have nest OHLC data
        #######
        
        self.compute_weightedUG()
        self.compute_backtestFixing()
        self.compute_rebalFixing()
        self.compute_recalibFixing()
        
        self.chkpt_dir = config.CPT_PATH 
        self.chkpt_dir = ''.join([config.CPT_PATH,'/bt'])
        self.checkpoint_file = os.path.join(self.chkpt_dir, 'backtesting.chkpt')
        
        self.irl_bt = irl_market.agent(df_input=self.df_input,saver=self.saver, session=self.sess)  
                    
    
    def _create_placeholders(self):
        
        print('Create placeholders')
        
        # Backtesting [None,self.tokenN]
        self.tokenW=tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tokenW')
        self.assetUnderMan = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='assetUnderMan')
        self.holdings = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='holdings')
        self.gasCost = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='gasCost')
        self.totalCost = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='totalCost')
        self.rebalanceCost = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='rebalanceCost')
        self.deltaPrice = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='deltaPrice')
        self.slippageCost =tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='slippageCost')
        self.phiValue = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='phiValue') 
        self.nextXt = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='nextXt')  # Tensor
        self.xt_nrm = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='xt_nrm')
    
    def compute_weightedUG(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            self.wUG = self.ugscores.mean(axis=0)

    def compute_rebalFixing(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if self.rebalFreq == '1D':
                self.rebalFixing = self.backtestFixing
            else:
                if self.rebalFreq == '1W':
                    self.rebalFixing = None
                if self.rebalFreq == '2W':
                    self.rebalFixing = None
   
    def compute_recalibFixing(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if self.recalibFreq == '1W':
                self.recalibFixing = pd.date_range(start=self.ldateBT, end=self.udateBT, freq='W').to_frame()
            elif self.recalibFreq == '1D':
                self.recalibFixing = pd.date_range(start=self.ldateBT, end=self.udateBT, freq='D').to_frame()
            else:
                self.recalibFixing = pd.date_range(start=self.ldateBT, end=self.udateBT).to_frame()
   
    def compute_backtestFixing(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            tmp = self.timeVec.loc[self.ldateBT:self.udateBT]
            #tmp = tmp[:-1]
            self.backtestFixing = tmp

    def f_nextXt(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            
            self.kappa = self.fitparams[0]
            self.sigma = self.fitparams[1]
            self.xt_nrm = tf.convert_to_tensor(self.df_input.iloc[0:self.ind,:]/self.df_input.mean()) ##DataFrame TxN
            self.wz = tf.multiply(self.Wdiag, tf.transpose(self.zprime, perm=[1, 2, 0]))  # NxK * TxNxK = TxNxK
            self.WZ = tf.reduce_sum(self.wz, axis=2)  # TxN #sum over the signals for each T and N
            self.scale = tf.slice(self.xt_nrm, [0, 0], [1, -1])  # 1xN # scale taken from first T slice
            self.WzScale = tf.multiply(self.scale, tf.compat.v1.cumprod(1 + self.WZ))  # TxN #rescale the signals and propagate geometric series from pct
            self.nextXt = self.xt_nrm * self.kappa *(self.WzScale- self.xt_nrm) + self.xt_nrm
            
            return self.nextXt[-1,0:self.tokenN]
        

    def f_tokenW(self):
       
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
  
            if self.timedf in self.rebalFixing.values: 
                tmp_equal = tf.ones(self.tokenN) / self.tokenN
                tmp_mom = self.f_nextXt() 
                tmp_mom = tf.divide(tmp_mom,tf.reduce_sum(tmp_mom))
                tmp_ug = tf.divide(self.wUG , self.wUG.sum())
                tmp_ECMA = self.contrib * tf.transpose(tf.stack([tmp_equal, tmp_mom, tmp_ug], axis=0))
                tmp = tf.transpose(tf.reduce_sum(tmp_ECMA,axis=1)) # tf.constant(tmp_ECMA)
                tmp = tf.broadcast_to(tmp,[1,self.tokenN])
            else:
                tmp=tf.broadcast_to(self.tokenW[-1,:],[1,self.tokenN])
            
            self.tokenW=tf.concat([self.tokenW,tmp],0)
            return self.tokenW

    def f_holdings(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if self.timedf in self.rebalFixing.values:
                tmp1 = self.tokenW[-1,:] * self.assetUnderMan[-1,:]
                tmp2 = tf.compat.v1.multiply(self.openPrice.iloc[self.ind + 1, 0:self.tokenN], self.f_phiValue())
                tmp2= tmp2 * tf.reduce_sum(tf.abs(self.tokenW[-1,:]),axis=0)  ####
                tmp = tmp1 / tmp2
            else:
                tmp = self.holdings[-1,:]
            tmp = tf.broadcast_to(tmp,[1,self.tokenN])
            self.holdings = tf.concat([self.holdings,tmp],0)
            return self.holdings

    def f_assetUnderMan(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
           
            holdingUpdate = self.holdings[-2,:] * self.phiValue * (self.deltaPrice[-1,:] - self.totalCost[-1,:])
            tmp = self.assetUnderMan[-1,:] + tf.reduce_sum(holdingUpdate,axis=0) 
            tmp = tf.broadcast_to(tmp,[1,1])
            self.assetUnderMan = tf.concat([self.assetUnderMan,tmp],0)
            return self.assetUnderMan

    def f_deltaPrice(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if self.timedf in self.rebalFixing.values:
                tmp = self.closePrice.iloc[self.ind, 0:self.tokenN] - self.openPrice.iloc[self.ind - 1,
                                                                      0:self.tokenN]  #### openPrice in t-1 instead of t
            else:
                tmp = self.closePrice.iloc[self.ind, 0:self.tokenN] - self.closePrice.iloc[self.ind - 1, 0:self.tokenN]
            tmp = tf.broadcast_to(tmp,[1,self.tokenN])
            self.deltaPrice = tf.concat([self.deltaPrice,tmp],0)
            return self.deltaPrice

    def f_rebalanceCost(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            tmp_ = tf.abs(self.holdings[-2,:]) * self.closePrice.iloc[self.ind, 0:self.tokenN] + tf.abs(
                self.holdings[-1,:]) * self.openPrice.iloc[self.ind + 1, 0:self.tokenN]
            tmp_ = tmp_ * self.phiValue * self.gasCost[-1,:]
            tmp = tf.reduce_sum(tmp_,axis=0)
            tmp = tf.broadcast_to(tmp,[1,1])
            self.rebalanceCost =tf.concat([self.rebalanceCost,tmp],0)
            return self.rebalanceCost

    def f_slippageCost(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            tmp_ = tf.abs(self.holdings[-2,:]) * self.closePrice.iloc[self.ind, 0:self.tokenN]
            tmp_ = tmp_ * self.phiValue * self.gasCost[-1,:]
            tmp =  tf.reduce_sum(tmp_,axis=0)
            tmp = tf.broadcast_to(tmp,[1,1])
            self.slippageCost= tf.concat([self.slippageCost,tmp],0)
            return self.slippageCost

    def f_gasCost(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            tmp = 0.00003
            tmp = tf.broadcast_to(tmp,[1,1])
            self.gasCost = tf.concat([self.gasCost,tmp],0)
            return self.gasCost

    def f_totalCost(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if self.includeTxCost:
                tmp1 = self.f_gasCost()
                tmp2 = self.f_rebalanceCost()
                tmp3 = self.f_slippageCost()
                tmp = tmp1[-1,:] + tmp2[-1,:] + tmp3[-1,:]
                tmp = tf.broadcast_to(tmp,[1,1])
            else:
                tmp=tf.zeros([1,1])
            self.totalCost = tf.concat([self.totalCost,tmp],0)
            return self.totalCost

    def f_phiValue(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            self.phiValue = tf.ones(self.tokenN)
            #self.phiValue = tf.concat(self.phiValue,tmp)
            return self.phiValue

   

    def dataIngestion(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            # self.openPrice=self.openclose[0,:,:]
            # self.closePrice=self.openclose[1,:,:]
            self.openPrice = self.openclose
            self.closePrice = self.openclose
    
    def _create_irlmarket(self):
        self.irl_bt.irl_mkt.build_graph()
    
    def _create_backtest(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            
            self.ind = self.timeVec.index.get_loc(self.timedf)
            
            if self.initCalc:
                oN_array = tf.ones([2,self.tokenN]) / self.tokenN
                o_array = tf.ones([2,1])
                z_array = tf.zeros([2,1])
                self.tokenW=oN_array
                self.assetUnderMan= o_array
                self.holdings = tf.concat([utils.init_tf(self.openPrice.iloc[self.ind - 2,0:self.tokenN],self.tokenN),utils.init_tf(self.openPrice.iloc[self.ind - 1,0:self.tokenN],self.tokenN)],axis=0) 
                self.totalCost = z_array
                self.gasCost=z_array
                self.rebalanceCost=z_array
                self.slippageCost=z_array
                self.deltaPrice = o_array*(self.closePrice.iloc[self.ind-1, 0:self.tokenN] - self.openPrice.iloc[self.ind - 2,0:self.tokenN])
            
            
           # self.irl_bt.irl_mkt.udate= str(self.timedf.date())
            
            if self.timedf in self.recalibFixing.values:
                self.fitparams, self.fitmeans, active_index, self.zprime, self.Wdiag = self.irl_bt.run_steps(udate=str(self.timedf.date()))
                
            if self.timedf in self.rebalFixing.values:
                self.irl_bt.irl_mkt._create_signals()
                self.zprime = self.irl_bt.irl_mkt.zprimet 
                
            self.tokenW,self.holdings,self.totalCost,self.deltaPrice,self.assetUnderMan=[self.f_tokenW(),self.f_holdings(), self.f_totalCost(),self.f_deltaPrice(), self.f_assetUnderMan()]
            
            return self.tokenW,self.holdings,self.totalCost,self.deltaPrice,self.assetUnderMan
    
    def build_graph(self):
        self._create_placeholders()
        self._create_backtest()
        self._create_irlmarket()

        
        
    