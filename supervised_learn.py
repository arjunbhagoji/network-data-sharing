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
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

def count_correct(y,y_hat):
    count_correct=0.0
    for i in range(len(y)):
        if y_hat[i]==y[i]:
            count_correct+=1
    print(count_correct/len(X_test))
    return count_correct/len(X_test)

def classify_get_performance(df_list, X_test, y_test, classifier,seed=None):
    performance_numbers=[]
    for item in df_list:
        X=item[0]
        y=item[1]
        if classifier == 'RF':
            clf = RandomForestClassifier(max_depth=3, random_state=seed)
        elif classifier == 'SVM':
#             clf = LinearSVC(C=1.0,max_iter=10000)
            clf = SGDClassifier(loss='hinge', penalty='l2',max_iter=10000,shuffle=True,random_state=seed)
        elif classifier == 'NB':
            clf = GaussianNB()
        clf.fit(X, y)
        y_test_predict=clf.predict(X_test)
        performance_numbers.append(count_correct(y_test,y_test_predict))
    return performance_numbers

if __name__ == "__main__":

    seed = 777
    rng = np.random.default_rng(seed)  # can be called without a seed
    
    config = configparser.ConfigParser()
    config.read_file(open('configs/default.cfg'))

    parser = argparse.ArgumentParser()
    
    # Input args
    if config['DATA']['input_type']=='CIC-IDS-2017':
        parser.add_argument('--day_name',type=str,default='Friday')
        parser.add_argument('--attack_type',type=str,default='DDoS')
    
    # Classification args
    parser.add_argument('--classifier', type=str, default='RF')
    
    # Federation args
    parser.add_argument('--num_agents', type=int,default=10)
    parser.add_argument('--maintain_ratio', dest='maintain_ratio', action='store_true')
    parser.add_argument('--seq_select', dest='seq_select', action='store_true')

    args = parser.parse_args()
    
    if config['DATA']['input_type']=='CIC-IDS-2017':
        input_file=config['DATA']['input_dir']+'/'+config['DATA']['input_type']+'/'+args.day_name+'-'+args.attack_type+'.csv'
    
    X_train, y_train, X_test, y_test = read_data(config['DATA']['input_type'], input_file, float(config['DATA']['train_test_split']),rng,attack_type=args.attack_type)

    if config['REMAP']['remap_mode']!='None':
        print("remapping ip addresses")
        ips_to_remap_list = ["192.168.1.1", "192.168.1.2"]
#         print(df.head(10))
        df = remap_ip_dataframe(df, ips_to_remap_list, args)
#         print(df.head(10)[["sa", "da", "sa_remapped", "da_remapped"]])
    
    if args.num_agents>1:
        df_list=distribute_dataframe_np(X_train, y_train, args.num_agents, args.maintain_ratio, args.seq_select, rng)
    else:
        df_list=[(X_train,y_train)]
        
    test_accs=classify_get_performance(df_list, X_test, y_test,args.classifier,seed)
    
    if config['DATA']['input_type']=='CIC-IDS-2017':
        output_dir_name=config['DATA']['output_dir']+'/'+config['DATA']['input_type']+'/'+args.day_name+'-'+args.attack_type
    if not os.path.exists(output_dir_name):
	    os.makedirs(output_dir_name)
    out_file_name=args.classifier
    if args.maintain_ratio:
        out_file_name+='_mr'
    if args.seq_select:
        out_file_name+='_seq'
    out_file_name=output_dir_name+'/'+out_file_name+'.txt'
    f = open(out_file_name, mode='a')
    f.write('{}, {}, {} \n'.format(args.num_agents,np.mean(test_accs),np.sqrt(np.var(test_accs))))
    f.close()

