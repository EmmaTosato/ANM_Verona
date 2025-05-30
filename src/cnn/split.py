# split.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def create_split(args):
    # Create output directory if it doesn't exist
    os.makedirs(args['output_dir'], exist_ok=True)

    # Load the labels CSV file
    df_labels = pd.read_csv(args['labels_path'])

    # Exclude specified subjects
    if args['to_exclude']:
        df_labels = df_labels[~df_labels['ID'].isin(args['to_exclude'])].reset_index(drop=True)

    # Filter for specified groups
    df_pair = df_labels[df_labels[args['label_column']].isin([args['group1'], args['group2']])].reset_index(drop=True)

    subjects = df_pair['ID'].values
    labels = df_pair[args['label_column']].values

    # Split train vs test
    train_subj, test_subj = train_test_split(
        subjects,
        stratify=labels,
        test_size=args['test_size'],
        random_state=args['seed']
    )

    # Initialize split column
    df_pair['split'] = ''

    # Assign test split
    df_pair.loc[df_pair['ID'].isin(test_subj), 'split'] = 'test'

    if args['validation_flag']:
        train_df = df_pair[df_pair['ID'].isin(train_subj)].reset_index(drop=True)

        train_subjects = train_df['ID'].values
        train_labels = train_df[args['label_column']].values

        # Split train further into train and val
        train_subj_final, val_subj = train_test_split(
            train_subjects,
            stratify=train_labels,
            test_size=args['val_size'],
            random_state=args['seed']
        )

        df_pair.loc[df_pair['ID'].isin(train_subj_final), 'split'] = 'train'
        df_pair.loc[df_pair['ID'].isin(val_subj), 'split'] = 'val'
    else:
        df_pair.loc[df_pair['ID'].isin(train_subj), 'split'] = 'train'

    # Print statistics
    print("\nTraining set label distribution:")
    train_counts = df_pair[df_pair['split'] == 'train'][args['label_column']].value_counts()
    print(f"Total size of the training set: {train_counts.sum()}")
    print(train_counts)

    if args['validation_flag']:
        print("\nValidation set label distribution:")
        val_counts = df_pair[df_pair['split'] == 'val'][args['label_column']].value_counts()
        print(f"Total size of the validation set: {val_counts.sum()}")
        print(val_counts)

    print("\nTest set label distribution:")
    test_counts = df_pair[df_pair['split'] == 'test'][args['label_column']].value_counts()
    print(f"Total size of the testing set: {test_counts.sum()}")
    print(test_counts)

    # Save the dataframe with split column
    out_csv = os.path.join(args['output_dir'], f"{args['group1']}_{args['group2']}_splitted.csv")
    df_pair.to_csv(out_csv, index=False)

    print(f"\nSplit file saved to: {out_csv}")


if __name__ == '__main__':
    config_path = "parameters/config_split.json"

    with open(config_path, "r") as f:
        args = json.load(f)

    create_split(args)
