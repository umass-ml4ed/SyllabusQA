"""
Combine raw data from qualtrics and prolific to create dataset.

python -m code.data_wrangling.create_dataset
"""

import pandas as pd
import random, string
import os
import sys
import re
import numpy as np
from transformers import AutoTokenizer

from code.utils.utils import set_random_seed, save_csv, SEED, BASE_DIR


COLS = ["id", "syllabus_name", "question_type", "question", "answer", "answer_span_1", "answer_span_2", "answer_span_3", "answer_span_4", "answer_span_5", "reasoning_step_1", "reasoning_step_2", "reasoning_step_3", "reasoning_step_4", "reasoning_step_5"]
question_types = ["yes/no", "single factual", "multi factual", "single reasoning", "multi reasoning", "summarization", "no answer"]
question_type_to_col_names = {
                    "yes/no" : ["question", "answer", "answer_span_1"],
                    "single factual" : ["question", "answer", "answer_span_1"],
                    "multi factual" : ["question", "answer", "answer_span_1", "answer_span_2", "answer_span_3", "answer_span_4", "answer_span_5"],
                    "single reasoning" : ["question", "answer", "reasoning_step_1"],
                    "multi reasoning" : ["question", "answer", "reasoning_step_1", "reasoning_step_2", "reasoning_step_3", "reasoning_step_4", "reasoning_step_5"],
                    "summarization" : ["question", "answer", "answer_span_1", "answer_span_2", "answer_span_3", "answer_span_4", "answer_span_5"],
                    "no answer" : ["question", "answer"]
                    }
question_type_to_col_ids = {
                    "yes/no" : [["Q41_1", "Q41_3", "Q41_2"], ["Q42_1", "Q42_3", "Q42_2"]],
                    "single factual" : [["Q43_1", "Q43_3", "Q43_2"], ["Q44_1", "Q44_3", "Q44_2"]],
                    "multi factual" : [["Q45_6", "Q45_12", "Q45_7", "Q45_8", "Q45_9", "Q45_10", "Q45_11"], ["Q46_6", "Q46_12", "Q46_7", "Q46_8", "Q46_9", "Q46_10", "Q46_11"]],
                    "single reasoning" : [["Q47_1", "Q47_3", "Q47_2"], ["Q48_1", "Q48_3", "Q48_2"]],
                    "multi reasoning" : [["Q49_6", "Q49_12", "Q49_7", "Q49_8", "Q49_9", "Q49_10", "Q49_11"], ["Q50_6", "Q50_12", "Q50_7", "Q50_8", "Q50_9", "Q50_10", "Q50_11"]],
                    "summarization" : [["Q51_6", "Q51_12", "Q51_7", "Q51_8", "Q51_9", "Q51_10", "Q51_11"], ["Q52_6", "Q52_12", "Q52_7", "Q52_8", "Q52_9", "Q52_10", "Q52_11"]],
                    "no answer" : [["Q53_1", "Q53_2"], ["Q54_1", "Q54_2"]]
                    }
question_type_to_col_names_screening = {
                    "multi reasoning" : ["question", "answer", "reasoning_step_1", "reasoning_step_2", "reasoning_step_3", "reasoning_step_4", "reasoning_step_5"],
                    "summarization" : ["question", "answer", "answer_span_1", "answer_span_2", "answer_span_3", "answer_span_4", "answer_span_5"]
                    }
question_type_to_col_ids_screening = {
                    "multi reasoning" : [["Q49_6", "Q49_12", "Q49_7", "Q49_8", "Q49_9", "Q49_10", "Q49_11"]],
                    "summarization" : [["Q51_6", "Q51_12", "Q51_7", "Q51_8", "Q51_9", "Q51_10", "Q51_11"]]
                    }


def load_df(filename, folder, nrows=None, filetype="excel"):
    path = os.path.join(folder, filename)
    if( filetype=="csv" ):
        df = pd.read_csv(path, encoding="utf-8", nrows=nrows, skiprows=[1,2])
    else:
        df = pd.read_excel(path, nrows=nrows, skiprows=[1])
    df.dropna(axis=0, how="all", inplace=True)
    
    return df
    

def get_key():
    key = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    return key


def findWholeWord(w):
    # https://stackoverflow.com/a/5320179/6156852
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search


def process_yes_no_qa(row):
    # Manually fix one yes/no QA included in human eval to yes/no style (swap answer and answer span mistake by annotator)
    if( row["id"] == "c0npO42Qfe1DiYab" ):
        orig_answer = row["answer"]
        row["answer"] = row["answer_span_1"]
        row["answer_span_1"] = orig_answer
    if( row["question_type"] == "yes/no" ):
        if( ( findWholeWord("yes")(row["answer"]) ) and ( findWholeWord("no")(row["answer"]) ) ):
            print(f"Processing error: Question of type yes/no has both yes and no in answer: {row['id']}, {row['answer']}")
        elif( findWholeWord("yes")(row["answer"]) != None ):
            row["answer"] = "Yes"
        elif( findWholeWord("no")(row["answer"]) != None ):
            row["answer"] = "No"
        else:
            print(f"Processing error: Question of type yes/no doesn't contain yes or no in answer: {row['id']}, {row['answer']}")
            # Set to nan to drop these QA later
            row["answer"] = np.nan
            
    return row


def process_no_answer_qa(row):
    if( row["question_type"] == "no answer" ):
        if( row["answer"] != "No/insufficient information" ):
            print(f"Processing error in answer of question type no answer: {row['id']}, {row['answer']}")
            # Manually checked all <10 errors can be safely replaced with intended "No/insufficient information"
            row["answer"] = "No/insufficient information"
    
    return row


def process_data(df, hit_type="long"):
    df_processed = pd.DataFrame(columns=COLS)
    for _, row in df.iterrows():
        for question_type in question_types:
            new_row = {}
            new_row["syllabus_name"] = row["syllabus_name"]
            new_row["question_type"] = question_type
            max_index = 1 if hit_type == "short" else sys.maxsize
            for col_ids in question_type_to_col_ids[question_type][:max_index]:
                new_row["id"] = get_key()
                col_names = question_type_to_col_names[question_type]
                for col_id, col_name in zip(col_ids, col_names):
                    new_row[col_name] = row[col_id]
                df_processed = pd.concat([df_processed, pd.DataFrame([new_row])], ignore_index=True)
    
    return df_processed


def sanity_check_meta_info(row):
    # Assert at least one answer span is present
    if( row["question_type"] == "yes/no" or row["question_type"] == "single factual"):
        if( row["num_tokens_answer_span_1"] == 0 ):
            print(f"Processing error: Answer span 1 is empty: {row['id']}")
            row["answer"] = np.nan

    # Assert at least two answer spans are present
    if( row["question_type"] == "multi factual" or row["question_type"] == "summarization"):
        if( row["num_tokens_answer_span_1"] == 0 ):
            print(f"Processing error: Answer span 1 is empty: {row['id']}")
            row["answer"] = np.nan
        if( row["num_tokens_answer_span_2"] == 0 ):
            print(f"Processing error: Answer span 2 is empty: {row['id']}")
            row["answer"] = np.nan

    # Assert at least one reasoning step is present
    if( row["question_type"] == "single reasoning"):
        if( row["num_tokens_reasoning_step_1"] == 0 ):
            print(f"Processing error: Reasoning step 1 is empty: {row['id']}")
            row["answer"] = np.nan

    # Assert at least two reasoning steps are present
    if( row["question_type"] == "multi reasoning"):
        if( row["num_tokens_reasoning_step_1"] == 0 ):
            print(f"Processing error: Reasoning step 1 is empty: {row['id']}")
            row["answer"] = np.nan
        if( row["num_tokens_reasoning_step_2"] == 0 ):
            print(f"Processing error: Reasoning step 2 is empty: {row['id']}")
            row["answer"] = np.nan
    
    return row


def main():
    set_random_seed(SEED)
    total_qa = 0
    df_all = pd.DataFrame(columns=COLS)
    for path, hit_type in [("mturk/invited_hit_first_half", "long"), ("mturk/invited_hit_second_half", "long"), ("mturk/long_hit", "long"), ("mturk/short_hit", "short"), ("prolific/long_hit", "long"), ("prolific/screening", "screening")]:
        print(f"Processing folder: {path}")
        if( hit_type == "screening" ):
            print('Ignoring ("prolific/screening", "screening") since QA per question type will become imbalanced')
            continue
        dir_name = os.path.join(BASE_DIR, path, "filtered_manual")
        dir_name_save = os.path.join(BASE_DIR, path, "processed")
        directory = os.fsencode(dir_name)
        file_list = sorted(os.listdir(directory))
        for file in file_list:
            filename = os.fsdecode(file)
            if( filename.endswith(".xlsx") ):
                df = load_df(filename, dir_name, filetype="excel")
                df = process_data(df, hit_type)
                total_qa += len(df)
                save_csv(df, filename.split(".")[0] + "_processed", dir_name_save)
                df_all = pd.concat([df_all, df], ignore_index=True)
        print("No of current QA: ", total_qa)
    
    # Shuffle dataset
    df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Check exactly 7 unique question types
    assert len(df_all["question_type"].unique().tolist()) == 7
    
    # Process yes/no questions
    df_all = df_all.apply(lambda row: process_yes_no_qa(row), axis=1)

    # Process no answer / adversarial questions
    df_all = df_all.apply(lambda row: process_no_answer_qa(row), axis=1)

    # Note: In create_split.py, we process "what is the name of the professor / instructor / TA" type questions when creating splits since test set split for human eval was fixed. Processing it now will create a new test set.

    # Drop 2 rows of yes/no type questions with non yes/no type QA, and 
    assert ( len(df_all) - len(df_all[df_all["answer"].notna()]) ) == 2, "Error: Not dropping only 2 QA"
    df_all = df_all[df_all["answer"].notna()]

    print("Total QA (note 8 QA of style 'what is the name of the professor / instructor / TA' will be dropped during train-val-test split later): ", len(df_all))

    # Sanity check required meta information is present
    df_all = df_all.fillna("")
    # Tokenize using LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    col_names = ["question", "answer"] + [ f"answer_span_{i}" for i in range(1,6) ] + [ f"reasoning_step_{i}" for i in range(1,6) ]
    for col_name in col_names:
        print(f"->Processing col_name: {col_name}")
        tokenized = tokenizer(df_all[col_name].tolist(), padding=False, truncation=False, add_special_tokens=False)
        num_tokens = [len(token_list) for token_list in tokenized["input_ids"]]
        df_all[f"num_tokens_{col_name}"] = num_tokens
    df_all = df_all.apply(lambda row: sanity_check_meta_info(row), axis=1)
    # Drop samples with no meta information
    assert ( len(df_all) - len(df_all[df_all["answer"].notna()]) ) == 1, "Error: Not dropping only 1 QA"
    df_all = df_all[df_all["answer"].notna()]
    df_all = df_all.loc[:, ~df_all.columns.str.startswith('num_tokens_')]

    # Save dataset
    dir_name_save_dataset = os.path.join(BASE_DIR, "dataset_complete")
    save_csv(df_all, "syllabus_qa_dataset", dir_name_save_dataset)  


if __name__ == '__main__':
    main()