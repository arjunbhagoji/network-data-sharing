import pandas as pd
import glob, ipaddress
import numpy as np, os
pd.set_option('display.max_colwidth', None)

import argparse
import configparser

from utils.data import read_data
from utils.ip_replace import remap_ip_dataframe
from utils.distribution import distribute_dataframe_np

from sklearn.ensemble import RandomForestClassifier

def count_correct(y,y_hat):
    count_correct=0.0
    for i in range(len(y)):
        if y_hat[i]==y[i]:
            count_correct+=1
    print(count_correct/len(X_test))
    return count_correct/len(X_test)

def classify_get_performance(df_list, X_test, y_test):
    performance_numbers=[]
    for item in df_list:
        X=item[0]
        y=item[1]
        clf = RandomForestClassifier(max_depth=3, random_state=0)
        clf.fit(X, y)
        y_test_predict=clf.predict(X_test)
        performance_numbers.append(count_correct(y_test,y_test_predict))
    return performance_numbers

if __name__ == "__main__":
    
    config = configparser.ConfigParser()
    config.read_file(open('configs/default.cfg'))

    parser = argparse.ArgumentParser()
    
    # Input args
    parser.add_argument('--input_file', type=str, default='data/CIC-IDS-2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

    # Federation args
    parser.add_argument('--num_agents', type=int,default=100)
    parser.add_argument('--maintain_ratio', dest='maintain_ratio', action='store_true')
    parser.add_argument('--seq_select', dest='seq_select', action='store_true')

    args = parser.parse_args()
    
    X_train, y_train, X_test, y_test = read_data(config['DATA']['input_type'], args.input_file, float(config['DATA']['train_test_split']))

    if config['REMAP']['remap_mode']!='None':
        print("remapping ip addresses")
        ips_to_remap_list = ["192.168.1.1", "192.168.1.2"]
#         print(df.head(10))
        df = remap_ip_dataframe(df, ips_to_remap_list, args)
#         print(df.head(10)[["sa", "da", "sa_remapped", "da_remapped"]])
    
    if args.num_agents>1:
        df_list=distribute_dataframe_np(X_train, y_train ,args.num_agents,args.maintain_ratio,args.seq_select)
    else:
        df_list=[df]
        
    test_accs=classify_get_performance(df_list, X_test, y_test)

