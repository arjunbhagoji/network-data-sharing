import numpy as np
import pandas as pd

def data_cleanup(X,y):
    inf_indices=np.where(X==np.inf)
    nan_indices=np.where(X!=X)
    mask = np.ones(len(X), dtype=bool)
    mask[np.union1d(inf_indices[0],nan_indices[0])] = False
    X_mod=X[mask,:]
    y_mod=y[mask]
    return X_mod,y_mod

def read_data(input_type,input_file,train_split,rng,attack_types=None):
    
    if input_type == 'CIC-IDS-2017':
        df=pd.read_csv(input_file,delimiter=',')
        df_np=df.to_numpy()
        df_np_perm=df_np[rng.permutation(len(df_np))]
        # Removing port from features
        X=df_np_perm[:,1:-2]
        y=df_np_perm[:,-1]

        y_mod=np.zeros(len(y))
        for i in range(len(attack_types)):
            y_mod[np.where(y==attack_types[i])]=i+1
    if input_type == 'nfdump-CIC-IDS-2017':
        df=pd.read_csv(input_file,delimiter=',')
        df_np=df.to_numpy()
        df_np_perm=df_np[rng.permutation(len(df_np))]
        # Removing index from features
        X=df_np_perm[:,1:-2]
        y=df_np_perm[:,-1]

        y_mod=np.zeros(len(y))
        for i in range(len(attack_types)):
            y_mod[np.where(y==attack_types[i])]=i+1
    elif input_type == 'CIC-IDS-2018':
        df=pd.read_csv(input_file,delimiter=',')
        df_np=df.to_numpy()
        df_np_perm=df_np[rng.permutation(len(df_np))]
        # Removing port, protocol and timestamp from features
        X=df_np_perm[:,3:-2]
        y=df_np_perm[:,-1]
        
        # Creating labeled data
        y_mod=np.zeros(len(y))
        for i in range(len(attack_types)):
            y_mod[np.where(y==attack_types[i])]=i+1
    elif input_type == 'iot':
        # To-do
        #input_dir = "../../data/benign_attack/nfcapd.*.csv"

        # input_data_file_list = glob.glob(input_dir)
        # li = []
        # print("reading dataframe")
        # for filename in input_data_file_list:
        #         df = pd.read_csv(filename, index_col=None, header=0)
        #         li.append(df)
        # df = pd.concat(li, axis=0, ignore_index=True)
        df = pd.read_csv(args.input_csv, index_col=None, header=0)
        # filter out the last few lines that are aggregate stats and not flow records
        df = df[~pd.isna(df["pr"])]
    
    X_clean,y_clean=data_cleanup(X,y_mod)
    
    train_len=int(train_split*len(X_clean))
    # Splitting into train and test
    X_train=X_clean[:train_len]
    X_test=X_clean[train_len:]

    y_train=y_clean[:train_len]
    y_test=y_clean[train_len:]
    
    train_mal_len=0
    for i in range(len(attack_types)):
        train_mal_len+=len(np.where(y_train==i+1)[0])

    test_mal_len=0
    for i in range(len(attack_types)):
        test_mal_len+=len(np.where(y_test==i+1)[0])
    
    print('Number of malicious samples in train set: %s' % train_mal_len)
    print('Total length of train set: %s' % len(y_train))
    
    print('Number of malicious samples in test set: %s' % test_mal_len)
    print('Total length of test set: %s' % len(y_test))
    
    return X_train, y_train, X_test, y_test
    
    

