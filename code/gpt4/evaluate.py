import pandas as pd
import hydra
from tqdm import tqdm
import openai

from code.utils.utils import set_random_seed, clean_str, save_csv
from code.utils.load_data import load_data
from code.gpt4.run_gpt4 import run_gpt4


def create_prompt(row):
    return f"The question is: {clean_str(row['question'])}\nThe answer is:"


def evaluate(df_test, configs):
    openai.log = "warning"
    client = openai.OpenAI()

    tqdm.pandas()
    # Create prompt 
    print(f"Creating prompts...")
    df_test["prompt"] = df_test.progress_apply(lambda row: create_prompt(row), axis=1)

    # Run GPT-4 using retrieval assistant
    print(f"Running model...")
    # Attach file to GPT-4 assistant for retrieval
    assistant = client.beta.assistants.create(
        name="syllabus-qa",
        instructions="You are a teaching assistant. Answer questions from students on course logistics using the attached course syllabus document in your knowledge base. If an answer is not contained in the course syllabus output 'No/insufficient information'.",
        tools=[{"type": "retrieval"}],
        model="gpt-4-1106-preview"
        )
    df_test = df_test.progress_apply(lambda row: run_gpt4(row, assistant, client, configs), axis=1)

    return df_test


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)

    # Load data
    _train_set, _val_set, test_set = load_data(configs)
    end_index = len(test_set) if configs.end_index == -1 else configs.end_index
    test_set = test_set[configs.start_index : end_index]

    # Run GPT-4 model for evaluation
    df_test = pd.DataFrame(test_set)
    df_test = evaluate(df_test, configs)

    # Save evaluation responses
    filename = f"{configs.exp_name}_start-index-{configs.start_index}_end-index-{end_index}_type-{configs.syllabi_type}"
    save_csv(df_test, filename, configs.results_dir)


if __name__ == '__main__':
    main()