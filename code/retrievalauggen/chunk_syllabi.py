import hydra
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_syllabi():
    syllabi = {}
    # Get syllabi
    filenames = glob.glob(f"./syllabi/syllabi_redacted/text/*.txt")
    for name in filenames:
        with open(name, "r", encoding="ISO-8859-1") as f:
            syllabi[name.split("/")[-1].split(".txt")[0]] = f.read()
    
    return syllabi


def chunk_syllabi(syllabi, configs):
    chunked_syllabi = {}
    # Note: Some chunks will have no overlap if a separator from the separator list chunks them below chunk_size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=configs.chunk_size, chunk_overlap=configs.chunk_overlap, length_function = len, is_separator_regex = False)
    for name, syllabus in syllabi.items():
        chunked_syllabi[name] = text_splitter.split_text(syllabus)
    
    return chunked_syllabi


@hydra.main(version_base=None, config_path="../finetune/", config_name="configs")
def main(configs):
    syllabi = load_syllabi()
    syllabi = chunk_syllabi(syllabi, configs)


if __name__ == '__main__':
    main()