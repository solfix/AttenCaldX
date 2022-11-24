###############################
# Author: Capas Peng          #
# Email: solfix123@163.com    #
###############################

import numpy as np
import pandas as pd
import re

blosum50_aa = {
    'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
    'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
    'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
    'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
    'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
    'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
    'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
    'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
    'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
    'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
    'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
    'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
    'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
    'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
    'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
    'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
    'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
    'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
    'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
    'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5))
    }

pca15_aaidx = {
    'A': np.array([
        -0.97090603, -0.32368135, 15.72065182, -0.50884066,  3.74021858,
        -0.77879681,  3.38667656, -0.91306275,  3.00614751, -2.32919906,
         0.7870219 , -1.87437966,  1.53335388,  1.344461  ,  3.35815929]),
    'R': np.array([
         8.53814625, -13.78745836,  -4.7706284 ,  -1.16449378,
        -9.46561383,   6.79244266,   1.98971004,  -5.15274046,
        -2.93772088,  -5.89206951,   5.26993696,   1.61036613,
        -1.95208183,  -0.64409   ,   1.53934309]),
    'N': np.array([
        14.86976088,  1.57062419, -3.62321751,  5.63672398, -1.78396131,
        -3.62736822, -1.44652457,  5.39945262,  0.87616878,  4.1128776 ,
         1.91381611,  2.4206327 , -3.81386422, -5.242596  ,  3.97131233]),
    'D': np.array([
        18.12626589, -2.14738155, -0.25216984,  2.31366107,  6.52406372,
        -4.88386469, -9.13828094, -0.71024376, -2.10295526, -1.27865374,
         1.48217398,  1.43032329, -4.04631611,  5.72388976, -1.78674913]),
    'C': np.array([
        -8.36918251e+00,  8.30319345e+00, -6.61966946e+00,  1.38734140e+01,
         8.59531324e+00,  9.79914818e+00,  1.26911187e+00, -4.63752901e+00,
        -9.83921566e-01,  4.36342436e+00,  2.34365256e+00,  3.43445840e-01,
         1.18290876e+00,  1.32558804e-02, -6.96621503e-01]),
    'E': np.array([
        12.03159874, -13.30121908,   8.27995382,  -2.1963886 ,
         8.48308971,  -2.02004094,  -4.58165905,  -2.68526505,
        -2.60424798,  -0.13228263,  -0.2394944 ,  -2.40393485,
          4.1802745 ,  -4.40432039,   0.23979065]),
    'Q': np.array([
         7.9210965 , -8.69666506, -0.59701694,  0.02046296,  0.93026397,
         2.56565474,  1.14554766,  0.50500247,  0.71467344,  0.04438606,
        -6.00954372,  3.89329376,  2.44305793, -2.76002278, -1.6987172 ]),
    'G': np.array([
        14.83920873, 19.24138906,  5.93438829,  5.66738401, -5.81883134,
        -8.78866739,  5.73914235, -4.72849218, -4.20333219, -1.56492845,
        -1.37450169, -0.61253836,  0.6617134 , -0.69921372, -1.40639534]),
    'H': np.array([
         0.68805439, -6.17956596, -6.80550604,  3.94746531,  1.33297594,
        -0.51211304,  4.73783089,  8.74722377, -3.12401236, -1.57598186,
         1.98977266, -7.78869789,  0.99852322,  1.37575734, -0.43120018]),
    'I': np.array([
        -20.34084677,   4.14384693,   3.86394279,  -3.40504491,
         -2.78992754,   1.46082574,  -4.07088528,   0.58387152,
         -3.41337954,   1.84625632,  -3.12082912,  -1.14873131,
         -4.65093114,  -1.64689363,   1.46078507]),
    'L': np.array([
        -17.63623772,  -0.35239597,  11.83924317,  -5.37540851,
         -1.15954659,  -1.97240237,   0.8551395 ,   0.5305574 ,
          2.1201762 ,   4.05167588,   7.06745699,   2.57547229,
          1.4547904 ,   1.28288187,  -0.26554788]),
    'K': np.array([
        11.70658851, -13.64488577,   2.13621858,  -2.01605285,
        -6.38068619,   0.81226081,   4.37857192,  -1.6496082 ,
         1.42540472,   8.6009414 ,  -2.9141596 ,  -1.63197862,
        -1.76542326,   2.93593542,  -2.78651305]),
    'M': np.array([
        -15.59654594,  -5.74596901,   1.05654898,   1.8258127 ,
          6.65236588,  -1.21996347,   6.87749571,   2.80480871,
         -1.90746691,  -3.59099156,  -4.0585946 ,   5.83997368,
         -2.03857504,   2.47274257,   1.55383649]),
    'F': np.array([
        -18.59572859,   0.92235509,  -3.31743268,  -2.52174562,
         -0.45125923,  -4.09153376,  -0.39932096,   2.4112244 ,
         -1.0836413 ,  -1.47072537,   3.19432868,   2.04374527,
          0.29441083,  -3.18840009,  -6.45603094]),
    'P': np.array([
        16.21704723,  15.09127036,  -8.72320617, -18.77750006,
         6.46478124,   4.56244976,   3.05557799,   0.15690408,
        -0.66929999,   0.7338994 ,   0.3953321 ,  -0.16512658,
        -0.43811542,   0.05203475,   0.50555428]),
    'S': np.array([
        11.85274099,  6.88460214,  2.96725951,  3.6290185 , -2.88619312,
         2.32382186, -1.33830204,  3.31676758,  6.26373175, -0.89789266,
         0.739414  ,  1.15021805,  1.36104635,  1.13305848,  1.16395609]),
    'T': np.array([
         4.53029865,  5.12654936,  1.43605965,  1.7684372 , -3.39334852,
         5.24360932, -3.36448143,  2.20062621,  6.52685307, -4.74609137,
        -2.03988324, -0.65055289,  0.19272884, -0.25439792, -3.25573314]),
    'W': np.array([
        -16.29768602,  -3.90769937, -13.79255488,  -0.84325799,
          2.33264713,  -8.04187276,   0.50186494,  -6.42108915,
          7.2327353 ,  -1.12521691,  -1.01561524,  -3.54186466,
         -1.9445611 ,  -0.81438463,   1.37250825]),
    'Y': np.array([
        -7.49926   ,   1.31004507, -12.07792238,  -1.16106141,
        -6.8946393 ,  -3.23103968,  -4.86248111,   0.68815174,
        -2.45473764,   1.48233974,  -1.42869414,   2.18166427,
         7.4468281 ,   3.0783195 ,   2.84874949]),
    'V': np.array([
        -16.01441319,   5.49304582,   7.34505768,  -0.71258533,
         -4.03171245,   5.60745006,  -4.73473404,  -0.44655998,
         -2.68117513,  -0.63176764,  -2.98159019,  -3.67133046,
         -1.09976811,   0.24198258,   0.76951331])}

def enc_list_bl_max_len(aa_seqs, aa_codes, max_seq_len):
    '''
    aa_codes of a list of amino acid sequences with padding 
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - aa_codes : dictionnary: key= AA, value= aa_codes
        - max_seq_len: common length for padding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        e_seq=np.zeros((len(seq),len(aa_codes["A"])))
        count=0
        for aa in seq:
            if aa in aa_codes:
                e_seq[count]=aa_codes[aa]
                count+=1
            else:
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)
                
        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(0,n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    return enc_aa_seq


def load_tcrs(fname, minlen=10, maxlen=20):
    '''
    load tcr sequences from file
    
    parameters:
        fname: data file name
        minlen: minimum length of TCR
        maxlen: maximum length of TCR
    '''
    pat=re.compile('[\\*_XB]')
    
    tcrs = [line.strip() for line in open(fname).readlines()]
    tcrs = [tcr for tcr in tcrs if len(tcr) > 0]
    tcrs = [tcr for tcr in tcrs if tcr.startswith('C') and tcr.endswith('F')]
    tcrs = [tcr for tcr in tcrs if len(pat.findall(tcr))==0]
    tcrs = [tcr for tcr in tcrs if len(tcr)>=minlen and len(tcr)<=maxlen]
    
    return tcrs

def load_tcrs2(fname, minlen=10, maxlen=20):
    '''
    load tcr sequences from file
    
    parameters:
        fname: data file name
        minlen: minimum length of TCR
        maxlen: maximum length of TCR
    '''
    pat=re.compile('[\\*_XB]')
    tcrs = list(pd.read_table(fname)['aaSeqCDR3'])
    tcrs = [tcr for tcr in tcrs if len(tcr) > 0]
    tcrs = [tcr for tcr in tcrs if tcr.startswith('C') and tcr.endswith('F')]
    tcrs = [tcr for tcr in tcrs if len(pat.findall(tcr))==0]
    tcrs = [tcr for tcr in tcrs if len(tcr)>=minlen and len(tcr)<=maxlen]
    
    return tcrs

def load_tcrs3(fname, minlen=10, maxlen=20):
    '''
    load tcr sequences from file
    
    parameters:
        fname: data file name
        minlen: minimum length of TCR
        maxlen: maximum length of TCR
    '''
    pat=re.compile('[\\*_XB]')
    df = pd.read_table(fname)
    tcrs = [(tcr, prop) for tcr, prop in zip(df.aaSeqCDR3, df.cloneFraction) if len(tcr) > 0]
    tcrs = [(tcr, prop) for tcr, prop in tcrs if tcr.startswith('C') and tcr.endswith('F')]
    tcrs = [(tcr, prop) for tcr, prop in tcrs if len(pat.findall(tcr))==0]
    tcrs = [(tcr, prop) for tcr, prop in tcrs if len(tcr)>=minlen and len(tcr)<=maxlen]

    return pd.DataFrame(tcrs, columns=["aaSeqCDR3", "cloneFraction"])