# network-data-sharing

To remap the ips and split, run ```replace_and_split.py```. It currently works the provided sample ```annotate_small.csv``` which can be changed using the flag ```--input_csv```. The ips are remapped at random but can use a predetermined set by using the flag ```--replace_ip_list``` which expects a csv file with a column headed as 'ip'.
