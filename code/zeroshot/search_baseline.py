import time
import pandas as pd
import hydra
from tqdm import tqdm
import pathlib
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from code.utils.utils import get_device, set_random_seed, sanitize_configs, compute_rouge_l_f1, save_csv
from code.finetune.batch_collator import CollateWraperSearchBaseline
from code.retrievalauggen.get_retrieved_text import get_retrieved_syllabi_chunks
from code.utils.load_data import load_data, get_test_data_loader_search_baseline
from code.utils.data_utils import get_targets_answers, post_process_predictions


def run_search_baseline(test_set, configs, device):
    # Get test data loader
    # Load retrieved syllabi chunks for RAG if applicable
    assert configs.rag, "RAG must be true for search baseline"
    assert configs.k == 1, "RAG must retrieve top-1 for search baseline"

    retrieved_syllabi_chunks = get_retrieved_syllabi_chunks(configs) if configs.rag else None
    test_loader = get_test_data_loader_search_baseline(test_set, CollateWraperSearchBaseline, configs, retrieved_syllabi_chunks)

    # Run batched inference
    start_time = time.time()
    predictions = []
    prompts = []

    with tqdm(test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("Batch {}".format(batch_num))
            prompts = prompts + batch["prompts"]
            outputs = batch["prompts"][:]
            predictions_batch = outputs
            predictions = predictions + predictions_batch
    test_time = time.time() - start_time

    # Compute metrics
    targets = get_targets_answers(test_set)
    rouge_l_f1, scores = compute_rouge_l_f1(targets, predictions, configs)
    
    df_test = pd.DataFrame(test_set)
    df_test["prompt"] = prompts
    df_test["predicted_answer"] = predictions
    df_test["rouge_l_f1"] = scores
    pathlib.Path(configs.results_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{configs.exp_name}_{configs.model_name.replace('/', '_').lower()}_rag-{configs.rag}_top-{configs.k}_prompt-style-{configs.prompt_style}"
    filepath = os.path.join(configs.results_dir, f"{filename}.txt")
    with open(filepath, "w") as f:
        print(f"zeroshot with model: {configs.model_name}")
        print(f"metrics/test/rouge_l_f1: {rouge_l_f1}")
        print(f"logs/test/time: {test_time}s")
        print(f"mean rouge_l_f1 grouped by question type:\n{df_test.groupby('question_type')['rouge_l_f1'].agg(['mean', 'count'])}")
        f.write(f"zeroshot with model: {configs.model_name}\n")
        f.write(f"metrics/test/rouge_l_f1: {rouge_l_f1}\n")
        f.write(f"logs/test/time: {test_time}s\n")
        f.write(f"mean rouge_l_f1 grouped by question type:\n{df_test.groupby('question_type')['rouge_l_f1'].agg(['mean', 'count'])}")

    # Save predictions
    save_csv(df_test, filename, configs.results_dir)


@hydra.main(version_base=None, config_path="../finetune/", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanitize configs
    configs = sanitize_configs(configs)
    # Get device
    device = get_device(configs)
    # Load data
    _train_set, _val_set, test_set = load_data(configs)
    run_search_baseline(test_set, configs, device)


if __name__ == '__main__':
    main()