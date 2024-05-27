#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:23:48 2022

@author: stephanethomas
""" 

# Launches the SmartBeta Backtest based on parameters and input of the config.py file

import argparse
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf

from model.SmartBetaIndex import SmartBetaIndex
import config
from data import data



def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    pass
      
    
def _get_data():
    pass

def train():
    pass

def backtest(load_checkpoint=False,initCalib=False):

    graph = tf.Graph()

    with graph.as_default():    
        DATA = data.get_DF(config._CALIB_DATA, setindex=True,indexName='timestamp')
        PRICES=data.get_DF(config._PRICE_DATA, setindex=True,indexName='timestamp')
        SmartBeta = SmartBetaIndex(df_input=DATA,df_ohlc=PRICES)
        SmartBeta.build_graph()
        saver=tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session(graph=graph) as sess:
        print('Running session')
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(SmartBeta.backtest_strategy(load_checkpoint=config._LOADCHECKPOINT,session=sess, saver=saver))
            
    
              
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'backtest', 'plot'},
                        default='backtest', help="mode. if not specified, it's in the train mode")

    args = parser.parse_args()

    if not os.path.isdir(config.PROCESSED_PATH):
        data.prepare_dataset()
    print('Data ready!')
    # create checkpoints folder if there isn't one already
    data.make_dir(config.CPT_PATH)

    if args.mode == 'train':
        train()
    elif args.mode == 'backtest':
        backtest()

if __name__ == '__main__':
    main()