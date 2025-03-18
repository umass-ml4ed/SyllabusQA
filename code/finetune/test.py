import wandb 
import time
import pandas as pd
import os
import pathlib
import hydra
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from code.utils.utils import compute_rouge_l_f1, save_csv, get_device, set_random_seed, sanitize_configs, compute_accuracy
from code.utils.data_utils import get_targets_answers, post_process_predictions, get_targets_question_types
from code.finetune.batch_collator import CollateWraperGenerativeTest
from code.utils.load_data import load_data, get_test_data_loader
from code.retrievalauggen.get_retrieved_text import get_retrieved_syllabi_chunks


def test(test_set, wandb_run_name, configs, device):
    dir_model = f"{configs.model_checkpoint_dir}/{configs.exp_name}/{wandb_run_name}/best_val_loss/lora_model/"
    configs.testing = True
    peft_config = PeftConfig.from_pretrained(dir_model)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=configs.load_in_8bit
        ) 
    hf_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
        )
    model = PeftModel.from_pretrained(hf_model, dir_model).to(device)
    # Merge adapters with base model for faster inference
    model = model.merge_and_unload()
    # Set model to eval mode
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(dir_model)
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
    reasoning_steps = []
    preds_question_types = []
    raw_predictions = []
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
            predictions_batch, preds_question_types_batch, reasoning_steps_batch, raw_predictions_batch = post_process_predictions(predictions_batch, batch["prompts_len"], configs)
            predictions = predictions + predictions_batch
            reasoning_steps = reasoning_steps + reasoning_steps_batch
            preds_question_types = preds_question_types + preds_question_types_batch
            raw_predictions = raw_predictions + raw_predictions_batch
    test_time = time.time() - start_time

    # Compute metrics
    df_test = pd.DataFrame(test_set)
    df_test["prompt"] = prompts
    df_test["raw_prediction"] = raw_predictions
    targets = get_targets_answers(test_set)
    rouge_l_f1, scores = compute_rouge_l_f1(targets, predictions, configs)
    question_type_acc = -1
    if( configs.add_question_type or configs.add_reasoning_steps ):
        targets_question_types = get_targets_question_types(test_set)
        question_type_acc = compute_accuracy(targets_question_types, preds_question_types)
        df_test["pred_question_type"] = preds_question_types
    if( configs.add_reasoning_steps ):
        df_test["reasoning_step"] = reasoning_steps

    df_test["predicted_answer"] = predictions
    df_test["rouge_l_f1"] = scores
    pathlib.Path(configs.results_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(configs.results_dir, f"{configs.exp_name}_{configs.model_name.replace('/', '_').lower()}_{wandb_run_name}.txt")
    with open(filepath, "w") as f:
        print(f"metrics/test/rouge_l_f1: {rouge_l_f1}")
        print(f"metrics/test/question_type_acc: {question_type_acc}")
        print(f"logs/test/time: {test_time}s")
        print(f"mean rouge_l_f1 grouped by question type:\n{df_test.groupby('question_type')['rouge_l_f1'].agg(['mean', 'count'])}")
        f.write(f"metrics/test/rouge_l_f1: {rouge_l_f1}\n")
        f.write(f"metrics/test/question_type_acc: {question_type_acc}\n")
        f.write(f"logs/test/time: {test_time}s\n")
        f.write(f"mean rouge_l_f1 grouped by question type:\n{df_test.groupby('question_type')['rouge_l_f1'].agg(['mean', 'count'])}")

    # Log metrics to weights and biases
    if( configs.log_wandb ):
        wandb.log({"metrics/test/rouge_l_f1": rouge_l_f1})
        wandb.log({"metrics/test/question_type_acc": question_type_acc})
        wandb.log({"logs/test/time": test_time})

    # Save predictions
    filename = f"{configs.exp_name}_{configs.model_name.replace('/', '_').lower()}_{wandb_run_name}"
    save_csv(df_test, filename, configs.results_dir)


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanitize configs
    configs = sanitize_configs(configs)
    # Get device
    device = get_device(configs)
    # Load data
    _train_set, _val_set, test_set = load_data(configs)
    test(test_set, configs.wandb_run_name, configs, device)


if __name__ == '__main__':
    main()