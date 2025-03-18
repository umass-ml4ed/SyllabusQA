import argparse
import os
import json
import numpy as np
import pandas as pd
import evaluate
from tqdm import tqdm

SEED = 21

def calc_sim(args):
    if args.metric == "rouge":
        metric = evaluate.load("rouge", seed=SEED)
    else:
        metric = evaluate.load("bertscore", seed=SEED)
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    print("Gathering pairs...")
    all_syllabi = test_df["syllabus_name"].unique()
    syllabus_to_pairs = {syllabus: {"src_questions": [], "target_questions": []} for syllabus in all_syllabi}
    for syllabus in tqdm(all_syllabi):
        questions = test_df[test_df["syllabus_name"] == syllabus]["question"]
        for i, src_question in enumerate(questions):
            for target_question in questions[i + 1:]:
                syllabus_to_pairs[syllabus]["src_questions"].append(src_question)
                syllabus_to_pairs[syllabus]["target_questions"].append(target_question)

    print("Computing similarity...")
    for syllabus_data in tqdm(syllabus_to_pairs.values()):
        if args.metric == "rouge":
            sim = metric.compute(
                predictions=syllabus_data["target_questions"],
                references=syllabus_data["src_questions"],
                use_stemmer=True, use_aggregator=False)["rougeL"]
        else:
            sim = metric.compute(
                predictions=syllabus_data["target_questions"],
                references=syllabus_data["src_questions"],
                model_type="microsoft/deberta-xlarge-mnli")["f1"]
        syllabus_data["sim"] = sim

    with open(f"question_sim_{args.metric}.json", "w") as f:
        json.dump(syllabus_to_pairs, f, indent=2, ensure_ascii=False)

def get_eval_overview(syllabus_data: dict):
    sim = np.array(syllabus_data["sim"])
    total_sim = sim.sum()
    duplicated = sim > 0.99
    num_duplicates = duplicated.sum()
    src_qs = np.array(syllabus_data["src_questions"])
    target_qs = np.array(syllabus_data["target_questions"])
    num_pairs = len(sim)
    num_questions = (1 + np.sqrt(1 + 8 * num_pairs)) / 2 # Solve for n in n(n-1)/2 = num_pairs with quadratic formula :)
    return total_sim, num_duplicates, num_pairs, num_questions

def eval(args):
    with open(f"question_sim_{args.metric}.json") as f:
        data = json.load(f)
    total_sim = 0
    total_duplicates = 0
    total_questions = 0
    total_pairs = 0
    for syllabus, syllabus_data in data.items():
        syll_sim, num_duplicates, num_pairs, num_questions = get_eval_overview(syllabus_data)
        total_sim += syll_sim
        total_duplicates += num_duplicates
        total_questions += num_questions
        total_pairs += num_pairs
        print(f"{syllabus} - Avg Sim: {syll_sim / num_pairs:.8f}, Portion Duplicated: {num_duplicates / num_questions:.8f}")
    print(f"Intra-Syllabus - Avg Sim: {total_sim / total_pairs:.8f}, Portion Duplicated: {total_duplicates / total_questions:.8f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calc_sim", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--metric", type=str, choices=["rouge", "bertscore"], default="rouge")
    parser.add_argument("--data_dir", type=str, default="../dataset_split")
    args = parser.parse_args()

    if args.calc_sim:
        calc_sim(args)
    if args.eval:
        eval(args)
   
if __name__ == "__main__":
    main()
