# AttenCaldX

This repository contains the code and the data to train **AttenCaldX** model.

​    Contact: solfix123@163.com

## Usage

### Python and essential packages

```
python         3.9.7
numpy          1.20.3
pandas         1.4.1
tensorflow     2.8.0
```

### Input file format

The input files are tsv files in the following format:

```
aaSeqCDR3	cloneFraction
CASSLFSWRHQLQETQYF	0.0416254036598493
CASSARSTGELFF	0.04157696447793326
CASSPRLATITYEQYF	0.03474165769644779
CASSQETGRVDGYTF	0.020705059203444565
CASSLPGPWANTGELFF	0.015527448869752422
......
```

The sequences in the `aaSeqCDR3` column are the top 1,000 most abundant TCRs.

You can use the following command to extract TCR and its frequency information from raw files:

```
python extractTCRs.py --input Example_raw_file.tsv
```

or:

```
python extractTCRs.py --input Example_raw_file.tsv --output Example_output_file.tsv --tcr_col aaSeqCDR3 --prop_col cloneFraction
```

### Cancer index prediction with pre-trained models

1. Prediction of a single TCR file:

    ```
    python predictTCRs.py --input Example_output_file.tsv
    ```

2. Prediction of all TCR files in a directory

   ```
   python predictFolder.py --folder data/Health/ --output myoutput.tsv
   ```

### Use of cancer index

Different cancer types have different cut-off values. . The reference thresholds of some  cancers are listed below:

|                Cancer                 | Threshold |
| :-----------------------------------: | :-------: |
| Non-small-cell lung carcinoma (NSCLC) |   0.565   |
|          Ovarian cancer (OV)          |   0.600   |
|   Early breast cancer (BRCA-early)    |   0.572   |
|      Urothelial carcinoma (UCC)       |   0.575   |

