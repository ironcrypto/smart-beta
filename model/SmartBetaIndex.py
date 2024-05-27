#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:57:50 2022

@author: stephanethomas
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model.SmartBetaBacktest import SmartBetaBacktest
from model.irl_market import irl_market
from data import data
import time
import config

#build the index - is this necessary?

class SmartBetaIndex(object):
    def __init__(self, df_input,df_ohlc):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        print('Initialize new SmartBetaModel')

        self.tokensIndex = config._TOKENINDEX
        self.tokensCalib = data.get_calibPortfolioList()
        self.contrib = np.fromstring(config._CONTRIBUTIONS,dtype=np.float32, sep=',')
        self.tokenN = len(self.tokensIndex)
        self.rebalFreq = config._REBALFREQ
        self.rebalDay = config._REBALDAY
        self.recalibFreq = config._RECALIBFREQ
        self.signals = config._SIGNALS
        self.ugscores=data.get_DF('UGSCORES',True,'score')
        self.df_input=df_input
        self.df_ohlc=df_ohlc
        
    def _create_placeholders(self):
        
        self.launch_date = tf.compat.v1.placeholder(dtype=tf.dtypes.string, shape=None, name='launch_date')
        self.index_version = tf.compat.v1.placeholder(dtype=tf.dtypes.string, shape=None, name='index_version')
        self.vertical = tf.compat.v1.placeholder(dtype=tf.dtypes.string, shape=None, name='vertical')
        self.index_name=tf.strings.join([self.vertical,'_',self.launch_date,'_',self.index_version])
    
    def _create_strategy(self):
        self.backtest = SmartBetaBacktest(SmartBeta=self,df_input=self.df_input,df_ohlc=self.df_ohlc)
        self.backtest.build_graph()
    
    def train_strategy(self,df_input):
        train=irl_market(df_input) 
        train.build_graph()
       
        
    def backtest_strategy(self,saver,session,load_checkpoint):
        self.backtest.span = len(self.backtest.backtestFixing)
        self.backtest.initCalc=True
        self.backtest.saver=saver
        self.backtest.sess=session
        self.backtest.irl_bt.irl_mkt.sess=session
        self.backtest.irl_bt.irl_mkt.saver=saver
        self.backtest.irl_bt.sess=session
        
            
        i = 0
        
        # Calibrate print out
        print("....Backtesting Calculating ...")
        
        if self.backtest.initCalib:
            self.backtest.timedf = self.backtest.backtestFixing.iloc[0][0]
            self.backtest.irl_bt.irl_mkt.udate= str(self.backtest.timedf.date())
            self.backtest.fitparams, self.backtest.fitmeans, active_index, self.backtest.zprimet,self.backtest.Wdiag = self.backtest.irl_bt.run_steps(udate=self.backtest.irl_bt.irl_mkt.udate, load_checkpoint=load_checkpoint) 
                
            # with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
        while i < self.backtest.span:
        
            if i>1: self.backtest.initCalc=False
            
            self.backtest.timedf = self.backtest.backtestFixing.iloc[i][0]
            print(''.join(['Calculating ', str(self.backtest.timedf),'...']))

            self.backtest.sess.run(self.backtest._create_backtest())
            i += 1
        
        print("...Saving Backtesting Results ....")
        # Delete first row that was needed for initilization
        self.backtest.holdings=tf.slice(self.backtest.holdings,[1,0],[self.backtest.holdings.shape[0]-1,self.backtest.holdings.shape[1]])
        self.backtest.totalCost=tf.slice(self.backtest.totalCost,[1,0],[self.backtest.totalCost.shape[0]-1,self.backtest.totalCost.shape[1]])
        self.backtest.deltaPrice=tf.slice(self.backtest.deltaPrice,[1,0],[self.backtest.deltaPrice.shape[0]-1,self.backtest.deltaPrice.shape[1]])
        self.backtest.assetUnderMan=tf.slice(self.backtest.assetUnderMan,[1,0],[self.backtest.assetUnderMan.shape[0]-1,self.backtest.assetUnderMan.shape[1]])
        
        
        # Dataframe
        #tmp = pd.DataFrame([], index=self.backtest.backtestFixing)
        
        # Save to CSV
        tmp=pd.DataFrame(self.backtest.sess.run(self.backtest.holdings), index=self.backtest.backtestFixing.index)
        tmp.to_csv(os.path.join(config.OUTPUT_PATH, "".join([str(round(time.time())),'_results_bt_holdings.csv'])))   

        tmp=pd.DataFrame(self.backtest.sess.run(self.backtest.totalCost), index=self.backtest.backtestFixing.index)
        tmp.to_csv(os.path.join(config.OUTPUT_PATH, "".join([str(round(time.time())),'_results_bt_totalCost.csv'])))

        tmp=pd.DataFrame(self.backtest.sess.run(self.backtest.deltaPrice), index=self.backtest.backtestFixing.index)
        tmp.to_csv(os.path.join(config.OUTPUT_PATH, "".join([str(round(time.time())),'_results_bt_deltaPrice.csv'])))

        tmp=pd.DataFrame(self.backtest.sess.run(self.backtest.assetUnderMan), index=self.backtest.backtestFixing.index)
        tmp.to_csv(os.path.join(config.OUTPUT_PATH, "".join([str(round(time.time())),'_results_bt_assetUnderMan.csv'])))
        
       # return self.irl_bt.sess.run(self.fitparams), self.irl_bt.sess.run(self.fitmeans), self.irl_bt.sess.run(self.holdings), self.irl_bt.sess.run(self.totalCost), self.irl_bt.sess.run(self.assetUnderMan), self.irl_bt.sess.run(self.deltaPrice), self.irl_bt.sess.run(self.tokenW)       
        
        
        
    def launch_strategy(self):
        pass
    
    def build_graph(self):
        self._create_placeholders()
        self._create_strategy()
     
        
    
        