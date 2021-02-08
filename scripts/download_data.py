import os, sys

outdir = "../data/benign_attack/"
fname = outdir + "data_link.csv"
links = list(map(str.strip, open(fname).readlines()))

for link in links:
    print(link)
    cmd = "wget -bqc --no-check-certificate %s -P %s" % (link, outdir)
    os.system(cmd)
