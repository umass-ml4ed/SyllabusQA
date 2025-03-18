import os
import argparse
import json
import os
import re
import evaluate
import pandas as pd
import numpy as np

from openai_api import get_batched_responses, get_assistant_responses

ASSISTANT_NAME = "syllabus-qa-metric"
ASSISTANT_INSTRUCTIONS = "You are a teaching assistant. Please evaluate potential answers to questions, and use the attached course syllabus if needed."


def normalize_question_types(df: pd.DataFrame):
    df["question_type"] = df["question_type"].str.lower().str.replace("-", " ")


def gpt4_prompt(question: str, answer_1: str, answer_2: str):
    question = question.replace("\n", " ").strip()
    answer_1 = answer_1.replace("\n", " ").strip()
    answer_2 = answer_2.replace("\n", " ").strip()
    return "Your job is to evaluate the similarity of different answers to a single question. " +\
        "You will be given a question asking for information regarding a specific college course. " +\
        "You will also be given two possible answers to that question, " +\
        "and will have to evaluate the claims in one answer against the other.\n\n" +\
        "Steps:\n" +\
        "1. List all of the atomic claims made by Answer 1. " +\
        "Note that an answer saying that there is no information counts as a single claim.\n" +\
        "2. Tell me which of those claims are supported by Answer 2.\n" +\
        "3. Summarize the results using the template \"Score: <num supported claims>/<num total claims>\". " +\
        "Ensure that both numbers are integers.\n\n" +\
        f"Question: {question}\n" +\
        f"Answer 1: {answer_1}\n" +\
        f"Answer 2: {answer_2}"


def gpt4_eval(df: pd.DataFrame, args):
    # Get prompts and responses from gpt4, cache results
    cache_file_postfix = args.model + ("_syll" if args.use_syllabus else "")
    cache_filename = f"qa_metric_cache_{cache_file_postfix}.json"
    if os.path.exists(cache_filename):
        with open(cache_filename) as f:
            print ("Loading cache...")
            cache = json.load(f)
    else:
        cache = {}
    prompts_and_syllabi = [
        (gpt4_prompt(row["question_x"], str(row[a1id]), str(row[a2id])), row["syllabus_name_x"])
        for _, row in df.iterrows()
        for a1id, a2id in [("answer_x", "answer_y"), ("answer_y", "answer_x")]
    ]
    uncached_prompts = [ps[0] for ps in prompts_and_syllabi if ps[0] not in cache]
    if args.use_syllabus:
        syllabi = [os.path.join(args.syllabus_dir, f"{ps[1]}.pdf") for ps in prompts_and_syllabi if ps[0] not in cache]
        responses = get_assistant_responses(
            uncached_prompts, syllabi, args.model, ASSISTANT_NAME, ASSISTANT_INSTRUCTIONS)
    else:
        responses = get_batched_responses(
            uncached_prompts, model=args.model, max_tokens=500, batch_size=20, temperature=0, use_parallel=True, show_progress=True)
    for prompt, response in zip(uncached_prompts, responses):
        cache[prompt] = response
    prompts = [ps[0] for ps in prompts_and_syllabi]
    responses = [cache[prompt] for prompt in prompts]
    with open(cache_filename, "w") as f:
        print(f"Saving cache... ({len(cache)} entries)")
        json.dump(cache, f)

    # Calculate scores based on results
    scores = []
    score_re = re.compile(r"Score: ([\d+\.]+)/([\d+\.]+)")
    for response in responses:
        match = score_re.search(response)
        if match:
            num = float(match.group(1))
            denom = float(match.group(2))
            if num > denom or num < 0 or denom <= 0:
                print("Bad response!\n", response)
                scores.append(0)
            else:
                scores.append(num / denom)
        else:
            print("Bad response!\n", response)
            scores.append(0)
    scores = np.array(scores).reshape(-1, 2)
    prompts = np.array(prompts).reshape(-1, 2)
    responses = np.array(responses).reshape(-1, 2)
    return prompts[:, 0], prompts[:, 1], responses[:, 0], responses[:, 1], scores[:, 0], scores[:, 1]


def findWholeWord(w):
    # https://stackoverflow.com/a/5320179/6156852
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search


def fix_yes_no_answers(row):
    if( row["question_type"] == "yes/no" ):
        if( ( findWholeWord("yes")(row["answer"]) ) and ( findWholeWord("no")(row["answer"]) ) ):
            print(f"Error: Question of type yes/no has both yes and no in answer: {row['id']}, {row['answer']}")
        elif( findWholeWord("yes")(row["answer"]) != None ):
            row["answer"] = "Yes"
        elif( findWholeWord("no")(row["answer"]) != None ):
            row["answer"] = "No"
        else:
            print(f"Error: Question of type yes/no doesn't contain yes or no in answer: {row['id']}, {row['answer']}")
    
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp_filename_1", type=str) # Ground truth test set
    parser.add_argument("--comp_filename_2", type=str) # Predictions set
    parser.add_argument("--out", type=str)
    parser.add_argument("--syllabus_dir", type=str, default="../syllabi_redacted/pdf/")
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--use_syllabus", action="store_true")
    parser.add_argument("--do_rouge", action="store_true")
    parser.add_argument("--do_bertscore", action="store_true")
    parser.add_argument("--do_gpt4", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--align_human_annotations", action="store_true")
    parser.add_argument("--human_annotations", type=str)
    parser.add_argument("--fix_yes_no_answers", action="store_true")
    args = parser.parse_args()

    comp_data_1 = pd.read_csv(args.comp_filename_1)
    if( args.num_samples == -1 ):
        comp_data_2 = pd.read_csv(args.comp_filename_2)
    else:
        comp_data_2 = pd.read_csv(args.comp_filename_2, nrows=args.num_samples)
    normalize_question_types(comp_data_1)
    normalize_question_types(comp_data_2)
    
    # Clean up prediction column names
    # If both predicted_answer and answer is presented in predictions, use predicted_answer
    if( ("predicted_answer" in comp_data_2.columns) and ("answer" in comp_data_2.columns) ):
        comp_data_2 = comp_data_2.drop('answer', axis=1)
        comp_data_2 = comp_data_2.rename(columns={"predicted_answer": "answer"})
    if( ("pred_question_type" in comp_data_2.columns) and ("question_type" in comp_data_2.columns) ):
        comp_data_2 = comp_data_2.drop('question_type', axis=1)
        comp_data_2 = comp_data_2.rename(columns={"pred_question_type": "question_type"})
    assert "answer" in comp_data_2.columns, "Error: answer col not present in comp_data_2 csv"

    # Fix yes-no question answers if required
    if( args.fix_yes_no_answers ):
        comp_data_2 = comp_data_2.apply(lambda row: fix_yes_no_answers(row), axis=1)
    
    # Align with human annotation file ids to compare with human eval results
    if( args.align_human_annotations ):
        df_human_eval = pd.read_csv(args.human_annotations)
        comp_data_2 = comp_data_2[comp_data_2["id"].isin(df_human_eval["id"])]

    merged = pd.merge(comp_data_1, comp_data_2, on="id")
    print(f"No of samples in merged: {len(merged)}")

    if args.do_rouge:
        print("Computing ROUGE...")
        # Bootstrapping aggregation has randomness in result, so set seed
        rouge_metric = evaluate.load("rouge", seed=args.seed)
        rougescore = np.array(rouge_metric.compute(predictions=merged["answer_y"], references=merged["answer_x"], use_stemmer=True, use_aggregator=False)["rougeL"])
        merged["rougescore"] = rougescore
        avg_rougescore = rouge_metric.compute(predictions=merged["answer_y"], references=merged["answer_x"], use_stemmer=True, use_aggregator=True)["rougeL"]
    else:
        rougescore = np.zeros((len(merged),))
        avg_rougescore = 0

    if args.do_bertscore:
        print("Computing BERTScore...")
        bertscore_metric = evaluate.load("bertscore", seed=args.seed)
        bertscore = np.array(bertscore_metric.compute(
            predictions=merged["answer_y"], references=merged["answer_x"], model_type="microsoft/deberta-xlarge-mnli")["f1"])
        merged["bertscore"] = bertscore
        avg_bertscore = bertscore.mean()
    else:
        bertscore = np.zeros((len(merged),))
        avg_bertscore = 0

    if args.do_gpt4:
        print("Computing GPT-4...")
        recall_prompts, precision_prompts, recall_responses, precision_responses, recall, precision = gpt4_eval(merged, args)
        f1 = 2 * ((recall * precision) / (recall + precision))
        f1[np.isnan(f1)] = 0 # Handle divide by zero
        merged["gpt4_recall_prompt"] = recall_prompts
        merged["gpt4_precision_prompt"] = precision_prompts
        merged["gpt4_recall_response"] = recall_responses
        merged["gpt4_precision_response"] = precision_responses
        merged["gpt4_recall"] = recall
        merged["gpt4_precision"] = precision
        merged["gpt4_f1"] = f1
        gpt4_precision = precision.mean()
        gpt4_recall = recall.mean()
        gpt4_f1 = f1.mean()
    else:
        recall = np.zeros((len(merged),))
        precision = np.zeros((len(merged),))
        f1 = np.zeros((len(merged),))
        gpt4_precision = 0
        gpt4_recall = 0
        gpt4_f1 = 0

    type_correct = merged["question_type_x"] == merged["question_type_y"]
    merged["type_correct"] = type_correct
    type_accuracy = type_correct.mean()

    if( args.do_gpt4 ):
        filename = f"{args.comp_filename_2.split('.csv')[0]}_do-gpt4_eval_metric"
    else:
        filename = f"{args.comp_filename_2.split('.csv')[0]}_eval_metric"
    with open(f"{filename}.txt", "w") as f:
        message = f"Overall - Rouge: {avg_rougescore:.8f}, BertScore: {avg_bertscore:.8f}, GPT-4 Precision: {gpt4_precision:.8f}, GPT-4 Recall: {gpt4_recall:.8f}, GPT-4 F1: {gpt4_f1:.8f}, Type Acc: {type_accuracy:.8f}"
        print(message)
        f.write(f"{message}\n")
        
        # Stratified across 7 question types
        print("\nStratified across 7 question types:")
        f.write("\nStratified across 7 question types:\n")
        for qt in merged["question_type_x"].unique():
            type_mask = merged["question_type_x"] == qt
            avg_rougescore = rougescore[type_mask].mean()
            avg_bertscore = bertscore[type_mask].mean()
            gpt4_precision = precision[type_mask].mean()
            gpt4_recall = recall[type_mask].mean()
            gpt4_f1 = f1[type_mask].mean()
            type_recall = type_correct[type_mask].mean()
            type_precision = type_correct[merged["question_type_y"] == qt].mean()
            message = f"{qt} ({type_mask.sum()} questions) - Rouge: {avg_rougescore:.8f}, BertScore: {avg_bertscore:.8f}, GPT-4 Precision: {gpt4_precision:.8f}, GPT-4 Recall: {gpt4_recall:.8f}, GPT-4 F1: {gpt4_f1:.8f}, Type Recall: {type_recall:.8f}, Type Precision: {type_precision:.8f}"
            print(message)
            f.write(f"{message}\n")
        
        # Stratified across 3 question classes: explicit, implicit, and insufficient info
        print("\nStratified across 3 question classes:")
        f.write("\nStratified across 3 question classes:\n")
        explicit = ["yes/no", "single factual", "multi factual"]
        implicit = ["single reasoning", "multi reasoning", "summarization"]
        insufficient_info = ["no answer"]
        q_classes = [("explicit", explicit), ("implicit", implicit), ("insufficient_info", insufficient_info)]
        for name, q_class in q_classes:
            type_mask = merged["question_type_x"].isin(q_class)
            avg_rougescore = rougescore[type_mask].mean()
            avg_bertscore = bertscore[type_mask].mean()
            gpt4_precision = precision[type_mask].mean()
            gpt4_recall = recall[type_mask].mean()
            gpt4_f1 = f1[type_mask].mean()
            type_recall = type_correct[type_mask].mean()
            type_precision = type_correct[merged["question_type_y"].isin(q_class)].mean()
            message = f"{name} ({type_mask.sum()} questions) - Rouge: {avg_rougescore:.8f}, BertScore: {avg_bertscore:.8f}, GPT-4 Precision: {gpt4_precision:.8f}, GPT-4 Recall: {gpt4_recall:.8f}, GPT-4 F1: {gpt4_f1:.8f}, Type Recall: {type_recall:.8f}, Type Precision: {type_precision:.8f}"
            print(message)
            f.write(f"{message}\n")

    if args.out:
        merged.to_csv(args.out, index=False)
    else:
        merged.to_csv(f"{filename}.csv", index=False)


if __name__ == "__main__":
    main()