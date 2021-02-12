import glob, pandas as pd, time, datetime
pd.set_option('mode.chained_assignment', None)
input_dir = "../../data/iot_data/benign_attack/nfcapd.*.csv"
input_data_file_list = glob.glob(input_dir)

li = []
for filename in input_data_file_list:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

# filter out the last few lines that are aggregate stats and not flow records
df = df[~pd.isna(df["pr"])]

def getTimestamp(time_str):
    return time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timetuple())
    
df['st_timestamp'] = df['ts'].apply(getTimestamp)
df['et_timestamp'] = df['te'].apply(getTimestamp)

df_mac2ip = pd.read_csv('../../data/iot_data/iot_mac2ip.csv', sep='\t')
df_mac2ip['fname'] = df_mac2ip['MAC'].apply(lambda x: x.replace(':', '') + '.csv')
mac2ip_list = df_mac2ip.values.tolist()
dfTemp = df
dfTemp['flow_tag'] = ""
for i in range(0, len(mac2ip_list)):
    fname = '../../data/iot_data/annotations/' + mac2ip_list[i][3]
    print(fname)
    ip_addr = mac2ip_list[i][1]
    df_annotation = pd.read_csv(fname)
    l = df_annotation.values.tolist()
    for i in range(0, len(l)):
        v = l[i]
        (st, et) = (v[0], v[1])
        tag = v[2]
        dfTemp['flow_tag'].loc[(df['st_timestamp'] >= st) & (df['st_timestamp'] < et) & ((df['sa'] == ip_addr) | (df['da'] == ip_addr)) ] = tag

def getFlowType(x):
    if x == "":
        return 0
    else:
        return 1
dfTemp["flow_type"] = dfTemp["flow_tag"].apply(getFlowType)
outfile = "../../data/annotated_data.csv"
dfTemp.to_csv(outfile)
