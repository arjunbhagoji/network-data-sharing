import numpy as np
import pandas as pd
import ipaddress



## assumes normally distributed IPs. Input: Standard deviation of the IP distance, # of IP addresses
def generate_ips_from_normal(num_ip, std_dev=10000):
    mean_ip = 2155905152 # 128.128.128.128
    print(mean_ip, num_ip, std_dev)
    ip_list = np.random.normal(mean_ip, std_dev, num_ip)
    ip_list_str = [ipaddress.ip_address(int(x)).__str__() for x in ip_list]
    return ip_list_str



def generate_ips(args, std_dev=1000):
    if args.remap_mode == "file":
        df = pd.read_csv(args.replace_ip_list, header=0)
        ip_list = list(df["ip"])
    elif args.remap_mode == "normal_dist":
        mean_ip = 2155905152
        num_ip = 10000
        ip_list = np.random.normal(mean_ip, std_dev, num_ip)
        ip_list = [ipaddress.ip_address(int(x)).__str__() for x in ip_list]
    return ip_list


## function to remap IP addresses in the dataset
## assumes df with 4-tuple fields named as sa, sp, da, dp  
## input: dataframe with the flow-level data
## output: df with two new columns. sa_remapped and da_remapped 


def remap_ip_dataframe(df, ips_to_remap_list, args, ip_std_dev=100000):
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


    ip_list = generate_ips(args, std_dev=ip_std_dev)
    len_ip_list = len(ip_list)
    ips_to_remap_dict = dict.fromkeys(ips_to_remap_list, 1)

    last_remap_idx = 0
    flow_map = {}

    df["sa_remapped"] = df.apply(lambda x: remap_ip(x, 'sa'), axis=1)
    df["da_remapped"] = df.apply(lambda x: remap_ip(x, 'da'), axis=1)
    return df