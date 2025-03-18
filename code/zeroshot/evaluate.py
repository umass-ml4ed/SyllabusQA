import time
import pandas as pd
import hydra
from tqdm import tqdm
import pathlib
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from code.utils.utils import get_device, set_random_seed, sanitize_configs, compute_rouge_l_f1, save_csv
from code.finetune.batch_collator import CollateWraperGenerativeTest
from code.retrievalauggen.get_retrieved_text import get_retrieved_syllabi_chunks
from code.utils.load_data import load_data, get_test_data_loader
from code.utils.data_utils import get_targets_answers, post_process_predictions


def evaluate_zero_shot(test_set, configs, device):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=configs.load_in_8bit
        )   
    # Need custom device map to load llama 70B
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_name,
        quantization_config=bnb_config,
        device_map=device_map
        )
    # Set model to eval mode
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Batched inference requires tokenizer padding side as left
    tokenizer.padding_side = "left"

    # Get test data loader
    # Load retrieved syllabi chunks for RAG if applicable
    retrieved_syllabi_chunks = get_retrieved_syllabi_chunks(configs) if configs.rag else None
    test_loader = get_test_data_loader(test_set, CollateWraperGenerativeTest, tokenizer, device, configs, retrieved_syllabi_chunks)

    # Run batched inference
    start_time = time.time()
    predictions = []
    prompts = []

    with tqdm(test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("Batch {}".format(batch_num))
            prompts = prompts + batch["prompts"]
            outputs = model.generate(
                **batch["inputs"],
                max_new_tokens = configs.max_new_tokens,
                do_sample = configs.do_sample,
                top_p = configs.top_p,
                top_k = configs.top_k
                )
            predictions_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions_batch, _preds_question_types, _reasoning_steps, _raw_predictions = post_process_predictions(predictions_batch, batch["prompts_len"], configs)
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
    evaluate_zero_shot(test_set, configs, device)


if __name__ == '__main__':
    main()