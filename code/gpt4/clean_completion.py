"""
Merge GPT-4 retrieval assistant results from two separate runs into a single dataframe and clean the predicted answers.
"""
import pandas as pd
import hydra
import re
from code.utils.utils import clean_str

from code.utils.utils import set_random_seed, load_df, save_csv


def clean_completion(text, question_type):
    # Remove prompt 
    split_text = text.split("The answer is:")
    if( len(split_text) > 1 ):
        text = split_text[1]
    else:
        text = split_text[0]
    # Remove citations added by GPT-4
    pattern = r"【\d+†source】"
    text = re.sub(pattern, "", text)
    # Remove whitespace around text
    text = clean_str(text)
    
    return text


@hydra.main(version_base=None, config_path="../finetune/", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)
    df_gpt4_1 = load_df("gpt4_retrieval_assistant_start-index-0_end-index-200_type-pdf", "./results/gpt4/")
    df_gpt4_2 = load_df("gpt4_retrieval_assistant_start-index-200_end-index-400_type-pdf", "./results/gpt4/")
    df_gpt4 = pd.concat([df_gpt4_1, df_gpt4_2])
    df_gpt4["predicted_answer"] = df_gpt4.apply(lambda row: clean_completion(row["predicted_answer_raw"], row["question_type"]), axis=1)

    save_csv(df_gpt4, "gpt4_retrieval_assistant_start-index-0_end-index-400_type-pdf", "./results/gpt4/")
    

if __name__ == '__main__':
    main()