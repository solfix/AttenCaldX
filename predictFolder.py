###############################
# Author: Capas Peng          #
# Email: solfix123@163.com    #
###############################

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import argparse
import sys
from glob import glob

sys.path.append("./python/")
from tcr_utils import load_tcrs2
from tcr_transformer import build_transformer_model

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, lower=False)
tokenizer.fit_on_texts('ARNDCEQGHILKMFPSTWYV')

model = build_transformer_model()
model.load_weights('models/transformer_weights.h5')

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Predict cancer index with TCRs.')
    parser.add_argument("--folder", dest="folder", type=str, help="The folder with TCR file in .tsv format.", required=True)
    parser.add_argument("--output", dest="out_file", type=str, help="The output file.", default="")
    args = parser.parse_args()
    
    tcr_samples = glob(os.path.join(args.folder, '*.tsv'))
    ca_scores = []
    for tcr_file in tcr_samples:
        tcrs = load_tcrs2(tcr_file)
        X = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(tcrs), maxlen=20)
        proba = model.predict(X)[:,1]
        ca_scores.append(np.mean(proba))
        
    if args.out_file!="":
        pd.DataFrame(list(zip(tcr_samples, ca_scores)), columns=["Sample", "CancerIndex"]).to_csv(args.out_file, sep="\t", index=False)
    else:
        print("%s\t%s" % ("Sample", "CancerIndex"))
        for i in range(len(ca_scores)):
            print("%s\t%f" % (tcr_samples[i], ca_scores[i]))
