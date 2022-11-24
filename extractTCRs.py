###############################
# Author: Capas Peng          #
# Email: solfix123@163.com    #
###############################

import pandas as pd
import argparse

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Extract TCRs from raw data file.')
    parser.add_argument("--input", dest="in_file", type=str, help="The input raw file in .tsv format.", required=True)
    parser.add_argument("--output", dest="out_file", type=str, help="The output file.", default="")
    parser.add_argument("--tcr_col", dest="tcr_col", type=str, help="The column name of TCR.", default="aaSeqCDR3")
    parser.add_argument("--prop_col", dest="prop_col", type=str, help="The column name of TCR frequency.", default="cloneFraction")
    args = parser.parse_args()
    
    df = pd.read_table(args.in_file)
    df = df[[args.tcr_col, args.prop_col]]
    df.columns = ["aaSeqCDR3", "cloneFraction"]
    df = df.sort_values(by="cloneFraction", ascending=False).iloc[:1000,:]
    df.reset_index(inplace=True, drop=True)
    
    if args.out_file=="":
        print("%s\t%s" % ("aaSeqCDR3", "cloneFraction"))
        for i in range(len(df)):
            print("%s\t%f" % (df.loc[i, "aaSeqCDR3"], df.loc[i, "cloneFraction"]))
    else:
        df.to_csv(args.out_file, sep="\t", index=False)
