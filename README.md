# ANM Verona

## Data
- `FCmaps` are the mean maps, one for each subject
- `FCmaps_augmented` contains 10 maps for each subject
- `FCmaps_processed` contains the processed maps, one for each subject
- `FCmaps_processed`_augmented contains the processed maps, 10 for each subject.

### Dataset containing
- ID of the subject
- The diagnosis (Group)
- Sex
- Age
- Education

The regressors:
- CDR_SB: disease gravity with a larger range
- MMSE: Mini Mental Status
- (CDR: same but smaller range)

The subject 4_S_5003 is removed

### GMM on CDR_SB
- Assigning each CDB_SB value to a cluster using GMM, removing NaN values of the CDR_SB before
- New column for the metadata dataframe

### Preprocessing
- Aligning the metadata to the raw_df
- Optional threshold for values below 0.1 and 0.2
- Masking with 2 masks

- Possible output dataset:
- `df_thr01_gm_masked`
- `df_thr02_gm_masked`
- `df_thr01_har_masked`
- `df_thr02_har_masked`
- `df_gm_masked`
- `df_har_masked`