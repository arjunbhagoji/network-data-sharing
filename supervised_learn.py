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
from sklearn.tree import DecisionTreeClassifier

from multiprocessing import Process, Manager


# Add functions to compute TPR, TNR etc. and prec./recall
def count_correct(y,y_hat):
    count_correct=0.0
    tp=0.0
    fp=0.0
    fn=0.0
    total_positives=len(np.where(y!=0)[0])
    total_negatives=len(np.where(y==0)[0])
    for i in range(len(y)):
        if y_hat[i]==y[i]:
            count_correct+=1
        if y_hat[i]==y[i] and y[i]!=0:
            tp+=1
        if y_hat[i]!=y[i] and y[i]!=0:
            fn+=1
        if y_hat[i]!=y[i] and y[i]==0:
            fp+=1
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    accuracy=count_correct/len(y)
    print('True positives: %s' % (tp))
    print('False positives: %s' % (fp))
    print('Recall: %s' % (recall))
    print('Precision: %s' %(precision))
    print('Accuracy: %s' % (accuracy))
    return accuracy,precision,recall

def classify_get_performance(item, i, X_test, y_test, classifier, return_dict, seed=None):
    print('Agent num: %s' % i)
    X=item[0]
    y=item[1]
    if i==0:
        print('Size of split:%s' % len(X))
    if classifier == 'RF':
        clf = RandomForestClassifier(max_depth=3, random_state=seed)
    elif classifier == 'SVM':
        clf = SGDClassifier(loss='hinge', penalty='l2',max_iter=10000,shuffle=True,random_state=seed)
    elif classifier == 'NB':
        clf = GaussianNB()
    elif classifier == 'DT':
        clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(X, y)
    y_test_predict=clf.predict(X_test)
    accuracy,precision,recall=count_correct(y_test,y_test_predict)
    return_dict[str(i)]=[accuracy,precision,recall]
    
    return

if __name__ == "__main__":

    seed = 777
    rng = np.random.default_rng(seed)  # can be called without a seed
    
    config = configparser.ConfigParser()
    config.read_file(open('configs/default.cfg'))

    parser = argparse.ArgumentParser()
    
    # Input args
    if 'CIC-IDS-2017' in config['DATA']['input_type'] or 'CIC-IDS-2018' in config['DATA']['input_type']:
        parser.add_argument('--day_name',type=str,default='Friday')
        parser.add_argument('--attack_types',nargs='+')
    
    # Classification args
    parser.add_argument('--classifier', type=str, default='RF')
    
    # Federation args
    parser.add_argument('--num_agents', type=int,default=1)
    parser.add_argument('--maintain_ratio', dest='maintain_ratio', action='store_true')
    parser.add_argument('--seq_select', dest='seq_select', action='store_true')

    args = parser.parse_args()
    
    if 'CIC-IDS-2017' in config['DATA']['input_type']:
        input_file=config['DATA']['input_dir']+'/'+config['DATA']['input_type']+'/'+args.day_name
        for item in args.attack_types:
            input_file+='_'+item
        input_file+='.csv'
    elif 'CIC-IDS-2018' in config['DATA']['input_type']:
        input_file=config['DATA']['input_dir_large']+'/'+args.day_name
        for item in args.attack_types:
            input_file+='_'+item
        input_file+='.csv'
    
    # Load data
    print('Loading data')
    X_train, y_train, X_test, y_test = read_data(config['DATA']['input_type'], input_file, float(config['DATA']['train_test_split']),rng,attack_types=args.attack_types)

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
        
    # Run classification
    manager = Manager()
    return_dict = manager.dict()
    
    print('Classifying data')
    processes=[Process(target=classify_get_performance, args=(item, i, X_test, y_test, args.classifier, return_dict, seed)) for i, item in enumerate(df_list)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    test_accs=[]
    test_precs=[]
    test_recalls=[]
    for k,v in return_dict.items():
        test_accs.append(v[0])
        test_precs.append(v[1])
        test_recalls.append(v[2])

#     test_accs,test_precs,test_recalls=classify_get_performance(df_list, X_test, y_test,args.classifier,seed)
    
    if 'CIC-IDS-2017' in config['DATA']['input_type'] or 'CIC-IDS-2018' in config['DATA']['input_type']:
        output_dir_name=config['DATA']['output_dir']+'/'+config['DATA']['input_type']+'/'+args.day_name
        for item in args.attack_types:
            output_dir_name+='_'+item
    if not os.path.exists(output_dir_name):
	    os.makedirs(output_dir_name)
    out_file_name=args.classifier
    if args.maintain_ratio:
        out_file_name+='_mr'
    if args.seq_select:
        out_file_name+='_seq'
    out_file_name=output_dir_name+'/'+out_file_name+'.txt'
    f = open(out_file_name, mode='a')
    if os.path.getsize(out_file_name) == 0:
        f.write('Num agents, acc_mean, acc_var, prec_mean, prec_var, recall_mean, recall_var \n')
    f.write('{}, {}, {}, {}, {}, {}, {} \n'.format(args.num_agents, np.mean(test_accs), np.sqrt(np.var(test_accs)), np.mean(test_precs), np.sqrt(np.var(test_precs)), np.mean(test_recalls), np.sqrt(np.var(test_recalls))))
    f.close()

