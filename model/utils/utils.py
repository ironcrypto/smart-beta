#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:48:50 2022

@author: stephanethomas
"""
import pandas as pd
import numpy as np
from model import *
import matplotlib.pyplot as plt
import tensorflow as tf


#checkpoint Loading and saving to allow stop and go of the runtime because of long training time        
def load_checkpoint(saver,sess,checkpoint_file):
    print('...loading checkpoint...')
    saver.restore(sess,checkpoint_file)
  
def save_checkpoint(saver,sess,checkpoint_file):
    print('...saving checkpoint...')
    saver.save(sess,checkpoint_file)
    
def init_tf(init_value,scale):
    weight=tf.ones([1,scale])/scale
    return weight/(tf.constant(init_value)*tf.reduce_sum(weight))
    
    
    
def plot_irl_check(df_ts,tokens,fitted_means,active_index,scale):
    
    # Note: tokens can be fewer than the training set but Dates must be equal because it impacts the average
    # via the active_index. Timeframe of plot could be made adjustable though, but we judged it a fancy feature.
    
    # Below: to account for the fact that we may have trained of the whole dataset but are passing on fewer tokens to plot
    if fitted_means.shape[1] == len(tokens): 
        mean_levels = pd.DataFrame(fitted_means,index=active_index,
                                       columns=tokens)
        # average market cap or price over the period used to normalize during training
        avg_mkt_cap_price = df_ts.loc[active_index,tokens].sum(axis=1).mean() 
        #avg_mkt_cap_price =
    
    else:
        mean_levels = pd.DataFrame(fitted_means,index=active_index,
                                       columns=df_ts.columns.values)
        # average market cap or price over the period used to normalize during training
        avg_mkt_cap_price = df_ts.loc[active_index,df_ts.columns.values].sum(axis=1).mean() 
    
    
    
    data=df_ts.loc[active_index,tokens]
    nplot = len(tokens)
    title = 'Market Cap '

    
    N = data.shape[1]

    if N > nplot: N = nplot

    plt.figure(figsize=(15,N))
    plt.suptitle(title + ' vs fitted mean reversion level',size=20)
    ytop = 0.96-0.4*np.exp(-N/5)
    plt.subplots_adjust(top=ytop)

    tokens = data.columns[:N]
    for index, token in enumerate(tokens,1):
        plt.subplot(int(np.ceil(N/3)),3,index)
        #plt.plot(1/scale*data.loc[active_index][token],color='blue',label='Market cap or price ($)')
        plt.plot(1/scale*data.loc[active_index][token].mean()*(mean_levels.loc[active_index][token]),color='red',label='Mean reversion level')
        plt.plot(1/scale*avg_mkt_cap_price*(mean_levels.loc[active_index][token]),color='red',label='Mean reversion level')
        plt.title(token,size=12)
        plt.xticks([])