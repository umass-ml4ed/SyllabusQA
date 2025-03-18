"""
Create train-val-test splits.

python -m code.data_wrangling.create_split
"""

import random
import os
import numpy as np
import pandas as pd

from code.data_wrangling.create_dataset import findWholeWord
from code.data_wrangling.anonymize_syllabi_folder import get_anonymized_syllabi_names
from code.utils.utils import set_random_seed, load_dataset, save_csv, SEED, BASE_DIR


def split_train_test(df):
    # Split on syllabi to ensure unseen syllabi in test set
    syllabi = df["syllabus_name"].unique().tolist()
    num_train_syllabi = int(0.8 * len(syllabi))
    train_syllabi = random.sample(syllabi, num_train_syllabi)
    print(f"No of test syllabi: {len(syllabi) - len(train_syllabi)}")

    df_train = df[df["syllabus_name"].isin(train_syllabi)]
    df_test = df[~df["syllabus_name"].isin(train_syllabi)]
    print(f"No of test samples: {len(df_test)}")

    # Check no missing samples
    assert ( len(df) == (len(df_train) + len(df_test)) ), "Error: Missing samples"
    # Check no overlap
    assert ( len(pd.Series(np.intersect1d(df_train["id"].values, df_test["id"].values))) == 0 ), "Error: Train-test overlap"

    # Shuffle splits
    df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df_train, df_test


def split_train_val(df):
    syllabi = df["syllabus_name"].unique().tolist()
    num_train_syllabi = int(0.8 * len(syllabi))
    train_syllabi = random.sample(syllabi, num_train_syllabi)
    print(f"No of train syllabi: {len(train_syllabi)}")
    print(f"No of val syllabi: {len(syllabi) - len(train_syllabi)}")

    df_train = df[df["syllabus_name"].isin(train_syllabi)]
    df_val = df[~df["syllabus_name"].isin(train_syllabi)]
    print(f"No of train samples: {len(df_train)}")
    print(f"No of val samples: {len(df_val)}")

    # Check no missing samples
    assert ( len(df) == (len(df_train) + len(df_val)) ), "Error: Missing samples"
    # Check no overlap
    assert ( len(pd.Series(np.intersect1d(df_train["id"].values, df_val["id"].values))) == 0 ), "Error: Train-val overlap"

    # Shuffle splits
    df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df_train, df_val


def save_splits(df_train, df_val, df_test):
    dir_name_save_splits = os.path.join(BASE_DIR, "dataset_split")
    save_csv(df_train, "train", dir_name_save_splits) 
    save_csv(df_val, "val", dir_name_save_splits)  
    save_csv(df_test, "test", dir_name_save_splits)  


def repair_filenames(df):
    df["syllabus_name"].replace(["BCH8016 Solid State Analysis (SYL) 012219 - revised_redacted", "Most Recent P132 Syllabus - Spring 2022_redacted"], ["BCH8016 Solid State Analysis (SYL) 012219 - revised", "Most Recent P132 Syllabus - Spring 2022"], inplace=True)

    return df


def filter_prof_name_questions(row):
    question = row["question"].lower()
    if( ("name" in question) and (("prof" in question) or ("inst" in question) or (findWholeWord("ta")(question))) and ("pronoun" not in question) ):
        print(f"Processing error in question of type 'what is the prof/instruction/TA name: {row['id']}, {row['question']}")
        # Set to nan to drop these QA later
        row["answer"] = np.nan
    
    return row


def anonymize_syllabi_names(df, anonymized_syllabi_names):
    df = df.replace({"syllabus_name": anonymized_syllabi_names})

    return df


def main():
    set_random_seed(SEED)
    df = load_dataset()
    df_train, df_test = split_train_test(df)
    df_train, df_val = split_train_val(df_train)

    # Issue: Two syllabi were known with two different filenames
    for df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        print(f"processing: {name}")
        if( "BCH8016 Solid State Analysis (SYL) 012219 - revised_redacted" in df['syllabus_name'].unique() ):
            print(f"found BCH8016 Solid State Analysis (SYL) 012219 - revised_redacted")
        if( "BCH8016 Solid State Analysis (SYL) 012219 - revised" in df['syllabus_name'].unique() ):
            print(f"found BCH8016 Solid State Analysis (SYL) 012219 - revised")
        if( "Most Recent P132 Syllabus - Spring 2022_redacted" in df['syllabus_name'].unique() ):
            print(f"found Most Recent P132 Syllabus - Spring 2022_redacted")
        if( "Most Recent P132 Syllabus - Spring 2022" in df['syllabus_name'].unique() ):
            print(f"found Most Recent P132 Syllabus - Spring 2022")
    df_train = repair_filenames(df_train)
    df_val = repair_filenames(df_val)
    df_test = repair_filenames(df_test)
    
    # Note: similar course syllabi syllabus-RESEC213-2023spring-yp and ResEcon-213-Fall-2022 are present, but both are present in train set, so no train-val or train-test leakage
    # Manually handle overlap in train-test due to file repair and merge of "BCH8016 Solid State Analysis (SYL) 012219 - revised_redacted" and "BCH8016 Solid State Analysis (SYL) 012219 - revised", some samples from this syllabus are in train, and some are in test, since human eval has been performed on test, shift remaning samples from train to test
    print("Number of QA in train before shift: ", len(df_train))
    print("Number of QA in val before shift: ", len(df_val))
    print("Number of QA in test before shift: ", len(df_test))
    print("Total QA before shift: ", len(df_train) + len(df_val) + len(df_test))
    # Add samples from df_train of "BCH8016 Solid State Analysis (SYL) 012219 - revised" to df_test
    df_test = pd.concat([df_test, df_train[df_train["syllabus_name"] == "BCH8016 Solid State Analysis (SYL) 012219 - revised"]])
    # Reshuffle test
    df_test = df_test.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # Drop samples from df_train of "BCH8016 Solid State Analysis (SYL) 012219 - revised" to df_test
    df_train = df_train[df_train["syllabus_name"] != "BCH8016 Solid State Analysis (SYL) 012219 - revised"]
    
    # Assert no overlap in syllabi post filename repair and manual shifting
    assert len(set(df_train['syllabus_name'].unique().tolist()).intersection(set(df_val['syllabus_name'].unique().tolist()))) == 0, f"Error: Overlap in train-val syllabi: {set(df_train['syllabus_name'].unique().tolist()).intersection(set(df_val['syllabus_name'].unique().tolist()))}"
    assert len(set(df_train['syllabus_name'].unique().tolist()).intersection(set(df_test['syllabus_name'].unique().tolist()))) == 0, f"Error: Overlap in train-test syllabi: {set(df_train['syllabus_name'].unique().tolist()).intersection(set(df_test['syllabus_name'].unique().tolist()))}"
    assert len(set(df_val['syllabus_name'].unique().tolist()).intersection(set(df_test['syllabus_name'].unique().tolist()))) == 0, f"Error: Overlap in val-test syllabi: {set(df_val['syllabus_name'].unique().tolist()).intersection(set(df_test['syllabus_name'].unique().tolist()))}"
    # Assert no overlap in samples
    assert ( len(pd.Series(np.intersect1d(df_train["id"].values, df_val["id"].values))) == 0 ), "Error: train-val overlap"
    assert ( len(pd.Series(np.intersect1d(df_train["id"].values, df_test["id"].values))) == 0 ), "Error: train-test overlap"
    assert ( len(pd.Series(np.intersect1d(df_val["id"].values, df_test["id"].values))) == 0 ), "Error: val-test overlap"
    print("Number of QA in train after shift: ", len(df_train))
    print("Number of QA in val after shift: ", len(df_val))
    print("Number of QA in test after shift: ", len(df_test))
    print("Total QA after shift: ", len(df_train) + len(df_val) + len(df_test))

    # Drop "what is the name of the professor / instructor / TA" type questions
    df_train = df_train.apply(lambda row: filter_prof_name_questions(row), axis=1)
    df_val = df_val.apply(lambda row: filter_prof_name_questions(row), axis=1)
    df_test = df_test.apply(lambda row: filter_prof_name_questions(row), axis=1)
    num_samples = len(df_train) + len(df_val) + len(df_test)
    # Drop 8 rows of questions acros train-val-test of style "what is the name of the professor / instructor / TA"
    df_train = df_train[df_train["answer"].notna()]
    df_val = df_val[df_val["answer"].notna()]
    df_test = df_test[df_test["answer"].notna()]
    assert ((num_samples - (len(df_train) + len(df_val) + len(df_test))) == 8), "Error: Not dropping only 8 QA"

    # Anonymize syllabi names
    anonymized_syllabi_names = get_anonymized_syllabi_names()
    df_train = anonymize_syllabi_names(df_train, anonymized_syllabi_names)
    df_val = anonymize_syllabi_names(df_val, anonymized_syllabi_names)
    df_test = anonymize_syllabi_names(df_test, anonymized_syllabi_names)
    
    print("\nFinal total number of syllabi: ", len(df_train['syllabus_name'].unique().tolist()) + len(df_val['syllabus_name'].unique().tolist()) + len(df_test['syllabus_name'].unique().tolist()))
    print("Final number of syllabi in train: ", len(df_train['syllabus_name'].unique().tolist()))
    print("Final number of syllabi in val: ", len(df_val['syllabus_name'].unique().tolist()))
    print("Final number of syllabi in test: ", len(df_test['syllabus_name'].unique().tolist()))
    print("\nFinal number of QA: ", len(df_train) + len(df_val) + len(df_test))
    print("Final total number of QA in train: ", len(df_train))
    print("Final number of QA in val: ", len(df_val))
    print("Final number of QA in test: ", len(df_test))

    save_splits(df_train, df_val, df_test)


if __name__ == '__main__':
    main()