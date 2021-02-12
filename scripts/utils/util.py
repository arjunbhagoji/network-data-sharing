import pandas as pd
import glob, ipaddress
import numpy as np, os
pd.set_option('display.max_colwidth', None)


## assumes normally distributed IPs. Input: Standard deviation of the IP distance, # of IP addresses
def generate_ips_from_normal(num_ip, std_dev=10000):
    mean_ip = 2155905152 # 128.128.128.128
    print(mean_ip, num_ip, std_dev)
    ip_list = np.random.normal(mean_ip, std_dev, num_ip)
    ip_list_str = [ipaddress.ip_address(int(x)).__str__() for x in ip_list]
    return ip_list_str



def generate_ips(num_ip, mode = "normal_dist", filename = None, std_dev=1000):
    if mode == "file":
        df = pd.read_csv(filename, header=0)
        ip_list = list(df["ip"])
    elif mode == "normal_dist":
        mean_ip = 2155905152
        ip_list = np.random.normal(mean_ip, std_dev, num_ip)
        ip_list = [ipaddress.ip_address(int(x)).__str__() for x in ip_list]
    return ip_list


## function to remap IP addresses in the dataset
## assumes df with 4-tuple fields named as sa, sp, da, dp  
## input: dataframe with the flow-level data
## output: df with two new columns. sa_remapped and da_remapped 


def remap_ip_dataframe(df, ips_to_remap_list, num_ip, ip_gen_mode='normal_dist', ip_std_dev=100000, iplist_filename=None):
    def remap_ip(x, field_name):
        (src_ip, src_port, dst_ip, dst_port) = (x["sa"], x["sp"], x["da"], x["dp"])
        
        ip_to_map = src_ip
        if field_name == "da":
            ip_to_map = dst_ip

        nonlocal last_remap_idx, flow_map
        flow_tuple = (src_ip, src_port, dst_ip, dst_port)
        if ip_to_map in ips_to_remap_dict:
            if flow_tuple in flow_map:
                ip_to_map = flow_map[flow_tuple]
            else:
                ip_to_map = ip_list[last_remap_idx % len_ip_list]
                flow_map[flow_tuple] = ip_to_map
                last_remap_idx += 1
        return ip_to_map


    ip_list = generate_ips(num_ip, mode=ip_gen_mode, filename=iplist_filename, std_dev=ip_std_dev)
    len_ip_list = len(ip_list)
    ips_to_remap_dict = dict.fromkeys(ips_to_remap_list, 1)

    last_remap_idx = 0
    flow_map = {}

    df["sa_remapped"] = df.apply(lambda x: remap_ip(x, 'sa'), axis=1)
    df["da_remapped"] = df.apply(lambda x: remap_ip(x, 'da'), axis=1)
    return df


def distribute_dataframe(df,num_agents,maintain_ratio,seq_select):
    rng = np.random.default_rng()
    list_of_dfs=[]
    assert len(df)>=num_agents
    if maintain_ratio:
        benign_df=df[df['flow_type']==0.0]
        mal_df=df[df['flow_type']==1.0]
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
    
    return list_of_dfs


if __name__ == "__main__":
    # args
    # distribution
    # num_agents, maintain_ratio, seq_select

    #input_dir = "../../data/benign_attack/nfcapd.*.csv"
    input_dir = "annotate_data_small.csv"
    
    input_data_file_list = glob.glob(input_dir)
    li = []
    print("reading dataframe")
    for filename in input_data_file_list:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    # filter out the last few lines that are aggregate stats and not flow records
    df = df[~pd.isna(df["pr"])]
    print("remapping ip addresses")
    ips_to_remap_list = ["192.168.1.1", "192.168.1.2"]
    print(df.head(10))
    df = remap_ip_dataframe(df, ips_to_remap_list, 10000)
    print(df.head(10)[["sa", "da", "sa_remapped", "da_remapped"]])

    # parameters
    num_agents=10
    maintain_ratio=True
    seq_select=False
    
    out_dir = "out_data/"
    if not os.path.exists(out_dir):
        os.mkdirs(out_dir)
    distributed_df_list=distribute_dataframe(df,num_agents,maintain_ratio,seq_select)
    for (i,split_df) in enumerate(distributed_df_list):
        outfile = out_dir + "split_%d.csv" % i
        split_df.to_csv(outfile)
