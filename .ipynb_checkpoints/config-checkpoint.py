#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:02:00 2022

@author: stephanethomas
"""


# define the core paramaters to BT 

DATA_PATH = '/Users/stephanethomas/myML/smart-beta-bt/data'
MCAP_FILE = 'nft_infra_coins_hist_mcaps.csv'
PRICE_FILE = 'nft_infra_coins_hist_prices.csv'
OHLC_FILE= 'nft_infra_coins_hist_ohlc.csv'
UGSCORE_FILE='ugscores_bt.csv'
OUTPUT_PATH = '/Users/stephanethomas/myML/smart-beta-bt/output'
PROCESSED_PATH = '/Users/stephanethomas/myML/smart-beta-bt/data/processed'
CPT_PATH = '/Users/stephanethomas/myML/smart-beta-bt/checkpoints'

# define the list of tokens in the index
_TOKENINDEX = ['aavegotchi', 'audius', 'decentraland', 'dodo', 'enjincoin', 'nftx', 'rarible', 'superfarm',
                 'superrare', 'the-sandbox']

# define the hyperparamter for model and backtesting
_CALIB_DATA='MCAP'
_PRICE_DATA='PRICES'
_CONTRIBUTIONS = '0.40, 0.5, 0.1'
_RECALIBFREQ=None
_REBALFREQ='1D'
_REBALDAY=None
_STARTDATE_BT='2022-03-01'
_ENDDATE_BT='2022-03-29'
_LOWERDATE_HIST='2021-09-02'
_UPPERDATE_HIST='2022-02-28'
_SIGNALS=['const','MACD_signal']
_WINDOWS=[0, 7]
_N_ACTIONS=1
_LOADCHECKPOINT=True
_INCLUDE_TXCOST=True