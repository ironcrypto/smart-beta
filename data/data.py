#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:03:34 2022

@author: stephanethomas
"""

import os
import pandas as pd
import numpy as np
import config

def get_rawdata():
    print("Loading data ...")
    RAW_PRICES=pd.read_csv(os.path.join(config.DATA_PATH,config.PRICE_FILE))
    RAW_MCAP=pd.read_csv(os.path.join(config.DATA_PATH,config.MCAP_FILE))
    RAW_OHLC=pd.read_csv(os.path.join(config.DATA_PATH,config.OHLC_FILE))
    UGSCORES=pd.read_csv(os.path.join(config.DATA_PATH,config.UGSCORE_FILE))
    
    return RAW_PRICES,RAW_MCAP,RAW_OHLC,UGSCORES
    

def prepare_dataset():
    
    print("Prepping data to be model-ready...")
    
    make_dir(config.PROCESSED_PATH)
    
    DF_PRICES, DF_MCAP, DF_OHLC, DF_UGSCORES = get_rawdata()
    
    
    # delete first column produced by the coingecko API
    DF_MCAP.drop(columns=DF_MCAP.columns[0], axis=1, inplace=True)
    DF_PRICES.drop(columns=DF_PRICES.columns[0], axis=1, inplace=True)
    DF_OHLC.drop(columns=DF_OHLC.columns[0], axis=1, inplace=True)
    
    
    
    # ---Applying Only on variables with NaN values
    for i in DF_MCAP.columns[DF_MCAP.isnull().any(axis=0)]:  
        DF_MCAP[i].fillna(DF_MCAP[i].mean(), inplace=True)
    
    for i in DF_PRICES.columns[DF_PRICES.isnull().any(axis=0)]:  
        DF_PRICES[i].fillna(DF_PRICES[i].mean(), inplace=True)
    
    for i in DF_OHLC.columns[DF_OHLC.isnull().any(axis=0)]:  
        DF_OHLC[i].fillna(DF_OHLC[i].mean(), inplace=True)
    
    # timestamp to date
    
    DF_MCAP['timestamp'] = pd.to_datetime(DF_MCAP['timestamp'], unit='ms')
    DF_PRICES['timestamp'] = pd.to_datetime(DF_PRICES['timestamp'], unit='ms')
    DF_OHLC['timestamp'] = pd.to_datetime(DF_OHLC['timestamp'], unit='ms')
       
    # make timestamp as index
    
    DF_MCAP.set_index('timestamp', inplace=True)
    DF_PRICES.set_index('timestamp', inplace=True)
    DF_OHLC.set_index('timestamp', inplace=True)
    DF_UGSCORES.set_index('score', inplace=True)

    DF_MCAP.to_csv(os.path.join(config.PROCESSED_PATH,'DF_MCAP.csv'))
    DF_PRICES.to_csv(os.path.join(config.PROCESSED_PATH,'DF_PRICES.csv'))
    DF_OHLC.to_csv(os.path.join(config.PROCESSED_PATH,'DF_OHLC.csv'))
    DF_UGSCORES.to_csv(os.path.join(config.PROCESSED_PATH,'DF_UGSCORES.csv'))
  
    
def get_DF(dataName, setindex=False, indexName='',dtype='float32'):
    tmp =('DF_',dataName,'.csv')
    DF = pd.read_csv(os.path.join(config.PROCESSED_PATH,''.join(tmp)))
    if setindex:
        DF.set_index(indexName, inplace=True)
    for c in DF.columns.values:
        DF[c]=DF[c].astype(dtype)
        
    return DF



def get_calibPortfolioList():
    DF=pd.read_csv(os.path.join(config.PROCESSED_PATH,'DF_MCAP.csv'))
    return DF.columns.values[1:]


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

if __name__ == '__main__':
    prepare_dataset()