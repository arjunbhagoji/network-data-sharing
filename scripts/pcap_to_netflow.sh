##########
## converts a pcap file to netflow csv file
## assumes nfpcapd and nfdump are installed. Link: https://github.com/phaag/nfdump
## the output fields are documented in "netflow.fields.md"
## a single flow may have multiple entries because of the flow timeout. Need to experiment w/ different timeout values
##########

#input pcap file 
pcap_file=$1

#creates a tmp directory wherein the intermediate output is stored
mkdir tmp

#reads pcap and converts it into netflow format which is the read by the nfdump tool
sudo nfpcapd -r $1 -l tmp/ -T all

# output directory wherein the data should be stored
outdir=$2

# creates the output directory 
if [ ! -d $outdir ]
then	
	mkdir -p $outdir
fi

# all the intermediate files generated using nfpcapd are read one-by-one and csv files are generated containing flow stats
for filename in tmp/*; do
	outfile=$(basename $filename)
	nfdump -r $filename -b -o extended -o csv > $outdir/$outfile.csv
done

# delete the intermediate directory
rm -rf tmp
