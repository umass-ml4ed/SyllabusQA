"""
Quality control checks used in dataset.
"""

import pandas as pd
import argparse
import profanity_check
import os 
import numpy as np

BASE_DIR = "./data/dataset_complete"
TEXT_COL_NAMES = COLS = ["question", "answer", "answer_span_1", "answer_span_2", "answer_span_3", "answer_span_4", "answer_span_5", "reasoning_step_1", "reasoning_step_2", "reasoning_step_3", "reasoning_step_4", "reasoning_step_5"]


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="syllabus_qa_dataset.csv", help="Filename for quality control check")
    parser.add_argument("--length_thresh", type=float, default=3, help="Profanity threshold")
    params = parser.parse_args()
    
    return params


def check_length(s, row, col_name, args):
    # Check for short questions or answers
    if( "answer_span" not in col_name ):
        if( (not (row["question_type"] == "yes/no" and col_name == "answer")) and (not (row["question_type"] == "no answer" and col_name == "answer")) ):
            if( len(s.split(" ")) < args.length_thresh ):
                return True
    

def check_profanity(s, row, col_name, args):
    # Check for profanity
    if( profanity_check.predict([s])[0] == 1 ):
        return True
    

def quality_control_check(df, args):
    checks = [(check_length, "check_length"), (check_profanity, "check_profanity")]
    for index, row in df.iterrows():
        for check, check_name in checks:
            for col_name in TEXT_COL_NAMES:
                if( not pd.isna(row[col_name]) and check(row[col_name], row, col_name, args) ):
                    print(f"Check ID: {row['id']} with failed check: {check_name} for {col_name}: {row[col_name]} ")


def load_df(filename, folder, nrows=None):
    path = os.path.join(folder, filename)
    df = pd.read_csv(path, encoding="utf-8", nrows=nrows)
    df.dropna(axis=0, how="all", inplace=True)
    print(f"Total QA read: {len(df)}")

    return df


def main():
    args = add_params()    
    df = load_df(args.filename, BASE_DIR)
    quality_control_check(df, args)


if __name__ == '__main__':
    main()