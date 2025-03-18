import hydra
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rank_bm25 import BM25Okapi
from pyserini.analysis import Analyzer, get_lucene_analyzer

from code.utils.utils import set_random_seed, load_df, clean_str, load_json, save_json, merge_dict
from code.retrievalauggen.chunk_syllabi import chunk_syllabi, load_syllabi


def get_retrieval_query(row, analyzer):
    query = clean_str(row["question"])
    query_tokens = analyzer.analyze(query)
    
    return query_tokens


def get_retrieved_text(row, syllabi_chunks_tokens, syllabi_chunks, analyzer, configs):
    query_tokens = get_retrieval_query(row, analyzer)
    syllabus_chunks_tokens = syllabi_chunks_tokens[row["syllabus_name"]]
    syllabus_chunks = syllabi_chunks[row["syllabus_name"]]
    bm25 = BM25Okapi(syllabus_chunks_tokens)
    doc_scores = bm25.get_scores(query_tokens)
    indices = np.argsort(doc_scores)[::-1][:configs.k]
    retrieved_syllabi_chunks = [syllabus_chunks[index] for index in indices]
    assert len(retrieved_syllabi_chunks) > 0, f"Error: No text retrieved for {row['id']}"

    return retrieved_syllabi_chunks
    

def tokenize_syllabi_bm_25(syllabi_chunks, analyzer):
    syllabi_tokens = {}
    # Tokenize syllabi for BM-25
    for name, syllabus_chunks in syllabi_chunks.items():
        syllabi_tokens[name] = [analyzer.analyze(chunk) for chunk in syllabus_chunks]
    
    return syllabi_tokens


def add_retrieved_text(df, analyzer, configs):
    # Chunk syllabi
    syllabi = load_syllabi()
    syllabi_chunks = chunk_syllabi(syllabi, configs)
    syllabi_chunks_tokens = tokenize_syllabi_bm_25(syllabi_chunks, analyzer)

    # Retrieve text for each query
    retrieved_text = {}
    for _index, row in tqdm(df.iterrows(), total=df.shape[0]):
        retrieved_text[row["id"]] = get_retrieved_text(row, syllabi_chunks_tokens, syllabi_chunks, analyzer, configs)

    return retrieved_text


def get_retrieved_syllabi_chunks(configs):
    # Default Lucene analyzer for English uses the Porter stemmer: https://github.com/castorini/pyserini/blob/master/docs/usage-analyzer.md
    analyzer = Analyzer(get_lucene_analyzer(stemmer="porter"))
    all_retrieved_text = []
    for name in ["train", "val", "test"]:
        filename = f"df_{name}_retrievalauggen_{configs.retriever_name}_top-{configs.k}_oracle-retriever-{configs.oracle_retriever}_chunk-size-{configs.chunk_size}_chunk-overlap-{configs.chunk_overlap}"
        filepath = Path(f"./data/retrievalauggen/{filename}.json")
        if filepath.is_file():
            retrieved_text = load_json(f"{filename}", "./data/retrievalauggen")
        else:
            df = load_df(f"{name}", configs.data_dir)
            retrieved_text = add_retrieved_text(df, analyzer, configs)
            save_json(retrieved_text, f"{filename}", "./data/retrievalauggen") 
        all_retrieved_text.append(retrieved_text)
    all_retrieved_text = merge_dict(all_retrieved_text)
    
    return all_retrieved_text


@hydra.main(version_base=None, config_path="../finetune/", config_name="configs")
def main(configs):
    set_random_seed(configs.seed)
    df = get_retrieved_syllabi_chunks(configs)


if __name__ == '__main__':
    main()