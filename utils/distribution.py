import numpy as np
import pandas as pd


def distribute_dataframe_np(X,y,num_agents,maintain_ratio,seq_select,rng,label_name='flow_type'):
    list_of_dfs=[]
    assert len(X)>=num_agents
    if maintain_ratio:
        benign_indices=np.where(y==0.0)
        mal_indices=np.where(y!=0.0)
        benign_df=X[benign_indices]
        mal_df=X[mal_indices]
        benign_df_labels=y[benign_indices]
        mal_df_labels=y[mal_indices]
        benign_bs=int(np.ceil(len(benign_df)/num_agents))
        mal_bs=int(np.ceil(len(mal_df)/num_agents))
        #print(benign_bs,mal_bs)
        if seq_select:
            for i in range(num_agents):
                benign_df_curr=benign_df[i*benign_bs:(i+1)*benign_bs]
                mal_df_curr=mal_df[i*mal_bs:(i+1)*mal_bs]
                benign_df_labels_curr=benign_df_labels[i*benign_bs:(i+1)*benign_bs]
                mal_df_labels_curr=mal_df_labels[i*mal_bs:(i+1)*mal_bs]
                X_curr=np.vstack([benign_df_curr,mal_df_curr])
                y_curr=np.vstack([benign_df_labels_curr,mal_df_labels_curr])
                list_of_dfs.append((X_curr,y_curr))
        else:
            benign_set=np.arange(len(benign_df))
            mal_set=np.arange(len(mal_df))
            for i in range(num_agents):
                #print(benign_set,mal_set)
                if len(benign_set)>benign_bs:
                    ben_indices_curr=np.sort(rng.choice(benign_set,benign_bs,replace=False))
                    #print(ben_indices_curr)
                    benign_set=np.setdiff1d(benign_set,ben_indices_curr)
                else:
                    ben_indices_curr=benign_set
                benign_df_curr=benign_df[ben_indices_curr]
                benign_df_labels_curr=benign_df_labels[ben_indices_curr]
                if len(mal_set)>mal_bs:
                    mal_indices_curr=np.sort(rng.choice(mal_set,mal_bs,replace=False))
                    mal_set=np.setdiff1d(mal_set,mal_indices_curr)
                else:
                    mal_indices=mal_set
                mal_df_curr=mal_df[mal_indices_curr]
                mal_df_labels_curr=mal_df_labels[mal_indices_curr]
                X_curr=np.vstack([benign_df_curr,mal_df_curr])
                y_curr=np.hstack([benign_df_labels_curr,mal_df_labels_curr])
                list_of_dfs.append((X_curr,y_curr))
    else:
        bs=int(np.ceil(len(X)/num_agents))
        if seq_select:
            for i in range(num_agents):
                X_curr=X[i*bs:(i+1)*bs]
                y_curr=y[i*bs:(i+1)*bs]
                list_of_dfs.append((X_curr,y_curr))
        else:
            all_set=np.arange(len(X))
            for i in range(num_agents):
                if len(all_set)>bs:
                    indices_curr=np.sort(rng.choice(all_set,bs,replace=False))
                    all_set=np.setdiff1d(all_set,indices_curr)
                else:
                    indices_curr=all_set
                X_curr=X[indices_curr]
                y_curr=y[indices_curr]
                list_of_dfs.append((X_curr,y_curr))
#     if save:
#         for (i,split_df) in enumerate(list_of_dfs):
#             outfile = args.output_dir + '/' + "split_%d.csv" % i
#             split_df.to_csv(outfile)
    
    return list_of_dfs

def distribute_dataframe(df,num_agents,maintain_ratio,seq_select,label_name='flow_type'):
    rng = np.random.default_rng()
    list_of_dfs=[]
    assert len(df)>=num_agents
    if maintain_ratio:
        benign_df=df[df[label_name]==0.0]
        mal_df=df[df[label_name]!=0.0]
        benign_bs=int(np.ceil(len(benign_df)/num_agents))
        mal_bs=int(np.ceil(len(mal_df)/num_agents))
        #print(benign_bs,mal_bs)
        if seq_select:
            for i in range(num_agents):
                benign_df_curr=benign_df[i*benign_bs:(i+1)*benign_bs]
                mal_df_curr=mal_df[i*mal_bs:(i+1)*mal_bs]
                df_curr=pd.concat([benign_df_curr,mal_df_curr])
                list_of_dfs.append(df_curr)
        else:
            benign_set=np.array(benign_df.index)
            mal_set=np.array(mal_df.index)
            for i in range(num_agents):
                #print(benign_set,mal_set)
                if len(benign_set)>benign_bs:
                    ben_indices_curr=np.sort(rng.choice(benign_set,benign_bs,replace=False))
                    #print(ben_indices_curr)
                    benign_set=np.setdiff1d(benign_set,ben_indices_curr)
                else:
                    ben_indices_curr=benign_set
                benign_df_curr=benign_df.loc[ben_indices_curr]
                if len(mal_set)>mal_bs:
                    mal_indices_curr=np.sort(rng.choice(mal_set,mal_bs,replace=False))
                    mal_set=np.setdiff1d(mal_set,mal_indices_curr)
                else:
                    mal_indices=mal_set
                mal_df_curr=mal_df.loc[mal_indices_curr]
                df_curr=pd.concat([benign_df_curr,mal_df_curr])
                list_of_dfs.append(df_curr)
    else:
        bs=int(np.ceil(len(df)/num_agents))
        if seq_select:
            for i in range(num_agents):
                df_curr=df[i*bs:(i+1)*bs]
                list_of_dfs.append(df_curr)
        else:
            all_set=np.array(df.index)
            for i in range(num_agents):
                if len(all_set)>bs:
                    indices_curr=np.sort(rng.choice(all_set,bs,replace=False))
                    all_set=np.setdiff1d(all_set,indices_curr)
                else:
                    indices_curr=all_set
                df_curr=df.loc[indices_curr]
                list_of_dfs.append(df_curr)
    if save:
        for (i,split_df) in enumerate(list_of_dfs):
            outfile = args.output_dir + '/' + "split_%d.csv" % i
            split_df.to_csv(outfile)
    
    return list_of_dfs