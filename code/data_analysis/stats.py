"""
Count num tokens distribution.

python -m code.data_analysis.stats
"""

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from code.utils.utils import load_dataset_post_filtering, load_df


def get_token_count(df):
    df = df.fillna("")
    # Tokenize using LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    col_names = ["question", "answer"] + [ f"answer_span_{i}" for i in range(1,6) ] + [ f"reasoning_step_{i}" for i in range(1,6) ]
    for col_name in col_names:
        print(f"\n->Processing col_name: {col_name}\n")
        tokenized = tokenizer(df[col_name].tolist(), padding=False, truncation=False, add_special_tokens=False)
        num_tokens = [len(token_list) for token_list in tokenized["input_ids"]]
        df[f"num_tokens_{col_name}"] = num_tokens
        print(df[f"num_tokens_{col_name}"].mask(df[f"num_tokens_{col_name}"] == 0).describe())
        print(f"\nquestion type stratified:\n")
        for question_type in df["question_type"].unique().tolist():
            print(f"question_type: {question_type}")
            mask = (df[f"num_tokens_{col_name}"] == 0) | (df[f"question_type"] != question_type)
            print(df[f"num_tokens_{col_name}"].mask(mask).describe())


def get_split_stats(df):
    get_token_count(df)


def get_overall_stats(df_all):
    print(f"Numer of unique syllabi: {len(df_all['syllabus_name'].unique().tolist())}")

    for question_type in ["yes/no", "single factual", "multi factual", "single reasoning", "multi reasoning", "summarization", "no answer"]:
        print(f"Number of questions of type {question_type}: {len(df_all[df_all['question_type'] == question_type])}: with percentage: {(len(df_all[df_all['question_type'] == question_type]))/len(df_all)}")

    explicit = ["yes/no", "single factual", "multi factual"]
    implicit = ["single reasoning", "multi reasoning", "summarization"]
    insufficient_info = ["no answer"]
    for category, name in [(explicit, "explicit"), (implicit, "implicit"), (insufficient_info, "insufficient_info")]:
        print(f"Number of questions in category {name}: {len(df_all[df_all['question_type'].isin(category)])}: with percentage: {(len(df_all[df_all['question_type'].isin(category)]))/len(df_all)}")


def get_dataset_split(row, train_syllabi, val_syllabi, test_syllabi):
    if( row["filename"] in train_syllabi ):
        row["dataset_split"] = "train"
    elif( row["filename"] in val_syllabi ):
        row["dataset_split"] = "val"
    elif( row["filename"] in test_syllabi ):
        row["dataset_split"] = "test"   
    else:
        print(f"Error: Syllabus not found {row['filename']}")

    return row


def get_num_tokens_syllabus(row):
    # Load syllabus in parsed as text file
    with open(f"./syllabi/syllabi_redacted/text/{row['filename']}.txt", "r", encoding="ISO-8859-1") as f:
        syllabi_text = f.read()
    # Tokenize using LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenized = tokenizer(syllabi_text, padding=False, truncation=False, add_special_tokens=False)
    row["num_tokens_syllabus"] = len(tokenized["input_ids"])

    return row
    

def get_syllabi_stats(df_train, df_val, df_test):
    df_syllabi = pd.read_csv("./syllabi/syllabi_meta_info.csv", encoding='unicode_escape')
    print(f"Total syllabi: {len(df_syllabi)}")
    print(f"No of unique areas: {len(df_syllabi['area'].unique())}")
    print(f"No of unique majors: {len(df_syllabi['major'].unique())}")
    print(f"No of unique universities: {len(df_syllabi['university'].unique())}")

    area_list = []
    for area in df_syllabi['area'].unique().tolist():
        area_list.append((area, round(len(df_syllabi[df_syllabi['area'] == area])/(len(df_syllabi)), 2)))
    print("sorted area list")
    area_list = sorted(area_list, key=lambda x: x[1], reverse=True)
    for area, percent in area_list:
        print(f"{area}, {percent*100}%")

    train_syllabi = df_train["syllabus_name"].unique().tolist()
    val_syllabi = df_val["syllabus_name"].unique().tolist()
    test_syllabi = df_test["syllabus_name"].unique().tolist()

    # Get train-val-test split mapping
    df_syllabi = df_syllabi.apply(lambda row: get_dataset_split(row, train_syllabi, val_syllabi, test_syllabi), axis=1)

    # Compute num pages distribution
    for split in ["train", "val", "test"]:
        print(f"Split: {split}")
        print(df_syllabi[df_syllabi["dataset_split"] == split]["num_pages"].describe())

    # Compute token distribution
    tqdm.pandas()
    print(f"Tokenizing syllabi")
    df_syllabi = df_syllabi.progress_apply(lambda row: get_num_tokens_syllabus(row), axis=1)

    print(df_syllabi["num_tokens_syllabus"].describe())
    for split in ["train", "val", "test"]:
        print(f"Split: {split}")
        print(df_syllabi[df_syllabi["dataset_split"] == split]["num_tokens_syllabus"].describe())    


def main(): 
    df_train, df_val, df_test = load_dataset_post_filtering()

    for df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        print(f"\n\nGetting stats for: {name}")
        get_split_stats(df)
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    get_overall_stats(df_all)

    get_syllabi_stats(df_train, df_val, df_test)

    # Distribution of q types in test set
    print(df_test["question_type"].value_counts()/len(df_test))
    explicit = ["yes/no", "single factual", "multi factual"]
    implicit = ["single reasoning", "multi reasoning", "summarization"]
    insufficient_info = ["no answer"]
    q_classes = [("explicit", explicit), ("implicit", implicit), ("insufficient_info", insufficient_info)]
    for name, q_class in q_classes:
        print(name)
        print(len(df_test[df_test["question_type"].isin(q_class)])/len(df_test))


if __name__ == '__main__':
    main()