import torch

from code.utils.utils import clean_str


def get_rag_context(item, retrieved_syllabi_chunks, configs):
    syllabi_chunks = retrieved_syllabi_chunks[item["id"]] 
    rag_context = "\nHere are snippets from the course syllabus which could be helpful in answering the question:"
    for i, chunk in enumerate(syllabi_chunks):
        rag_context += f"\n### Snippet {i+1}: {chunk}"
    rag_context += "\nThe relevant snippets from the course syllabus were added above."
    
    return rag_context
    

def get_system_prompt(configs):
    if( configs.add_question_type ):
        system_prompt = "<<SYS>>\nYou are a teaching assistant. Answer questions from students on course logistics. First, choose the question type from one of these 7 options: 1) yes/no, 2) single factual, 3) multi factual, 4) single reasoning, 5) multi reasoning, 6) summarization, and 7) no answer. Then, write the answer to the question.\n<</SYS>>"
    elif( configs.add_reasoning_steps ):
        system_prompt = "<<SYS>>\nYou are a teaching assistant. Answer questions from students on course logistics. First, choose the question type from one of these 7 options: 1) yes/no, 2) single factual, 3) multi factual, 4) single reasoning, 5) multi reasoning, 6) summarization, and 7) no answer. For questions of type single reasoning, provide a single reasoning step. For questions of type multi reasoning, provide up to 5 reasoning steps. For all other question types, don't provide any reasoning steps. Then, write the answer to the question.\n<</SYS>>"
    else:
        system_prompt = "<<SYS>>\nYou are a teaching assistant. Answer questions from students on course logistics.\n<</SYS>>"
        
    return system_prompt


def create_prompt(item, retrieved_syllabi_chunks, configs):
    rag_context = get_rag_context(item, retrieved_syllabi_chunks, configs) if configs.rag else ""
    system_prompt = get_system_prompt(configs)
    if( configs.add_question_type or configs.add_reasoning_steps ):
        # Remove full stop after question since question can end with "?"
        prompt = f"[INST] {system_prompt}{rag_context}\n### The question is: {clean_str(item['question'])}\n### The question type is: [/INST]"
    else:
        # Remove full stop after question since question can end with "?"
        prompt = f"[INST] {system_prompt}{rag_context}\n### The question is: {clean_str(item['question'])}\n### The answer is: [/INST]"

    return prompt


def create_completion(item, tokenizer, configs):
    # Suffix completion with tokenizer.eos_token
    if( configs.add_question_type ):
        completion = f""" {clean_str(item["question_type"])}\n### The answer is: {clean_str(item["answer"])}{tokenizer.eos_token}"""
    elif( configs.add_reasoning_steps ):
        reasoning_steps = get_reasoning_steps(item)
        completion = f""" {clean_str(item["question_type"])}{reasoning_steps}\n### The answer is: {clean_str(item["answer"])}{tokenizer.eos_token}"""
    else:
        completion = f""" {clean_str(item["answer"])}{tokenizer.eos_token}"""

    return completion


def get_reasoning_steps(item):
    reasoning_steps = ""
    if( item["question_type"] == "single reasoning" ):
        reasoning_steps += f"\n### Reasoning step 1: {clean_str(item['reasoning_step_1'])}"
    elif( item["question_type"] == "multi reasoning" ):
        for i in range(1, 6):
            if( item[f"reasoning_step_{i}"] != None ):
                reasoning_steps += f"\n### Reasoning step {i}: {clean_str(item[f'reasoning_step_{i}'])}"
    else:
        pass
    
    return reasoning_steps


class CollateWraperGenerative():
    def __init__(self, tokenizer, device, configs, retrieved_syllabi_chunks):
        self.tokenizer = tokenizer
        self.ignore_index = -100 # Default ignore index in CrossEntropyLoss
        self.device = device
        self.configs = configs
        self.retrieved_syllabi_chunks = retrieved_syllabi_chunks


    def __call__(self, batch):
        # Construct text
        prompts = [create_prompt(item, self.retrieved_syllabi_chunks, self.configs) for item in batch]
        examples = [f"{create_prompt(item, self.retrieved_syllabi_chunks, self.configs)}{create_completion(item, self.tokenizer, self.configs)}" for item in batch]

        # Tokenize
        # We assume no input is truncated by setting max length to be above expected max input sequence length based on dataset token count stats
        prompts_tokenized = self.tokenizer(prompts, padding=False, truncation=True, max_length=self.configs.max_length, add_special_tokens=True)
        examples_tokenized = self.tokenizer(examples, padding=True, truncation=True, max_length=self.configs.max_length, return_tensors='pt', add_special_tokens=True).to(self.device)

        # Construct labels
        labels = examples_tokenized["input_ids"].detach().clone()
        # Ignore pad tokens when computing loss
        labels = labels.masked_fill((examples_tokenized["attention_mask"] == 0), self.ignore_index)
        # Ignore prompt tokens when computing loss
        prompts_len = torch.tensor([len(prompt_tokenized_input_ids) for prompt_tokenized_input_ids in prompts_tokenized["input_ids"]]).to(self.device)
        range_tensor = torch.arange(examples_tokenized["input_ids"].size(1), device=self.device).unsqueeze(0)
        range_tensor = range_tensor.repeat(prompts_len.size(0), 1)
        mask_tensor = (range_tensor < prompts_len.unsqueeze(-1)) 
        labels[mask_tensor] = self.ignore_index
        
        if( self.configs.debug ):
            print(f"prompts: {prompts}")
            print(f"examples: {examples}")
            print(f"prompts_tokenized: {prompts_tokenized}")
            print(f"examples_tokenized: {examples_tokenized}")
            print(f"labels: {labels}")
            for ids in examples_tokenized["input_ids"]:
                print(self.tokenizer.decode(ids))

        return {
            "input_ids": examples_tokenized["input_ids"].to(self.device),
            "attention_mask": examples_tokenized["attention_mask"].to(self.device),
            "labels": labels.to(self.device)
            }


class CollateWraperGenerativeTest():
    def __init__(self, tokenizer, device, configs, retrieved_syllabi_chunks):
        self.tokenizer = tokenizer
        self.device = device
        self.configs = configs
        self.retrieved_syllabi_chunks = retrieved_syllabi_chunks


    def __call__(self, batch):
        assert self.tokenizer.padding_side == "left", "Batched inference requires tokenizer padding side as left"
        # Construct text
        prompts = [create_prompt(item, self.retrieved_syllabi_chunks, self.configs) for item in batch]
        prompts_len = [len(prompt) for prompt in prompts]
        # Tokenize
        prompts_tokenized = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.configs.max_length, return_tensors="pt", add_special_tokens=True)

        return {
            "inputs": prompts_tokenized.to(self.device),
            "prompts_len": prompts_len,
            "prompts": prompts
            }


class CollateWraperSearchBaseline():
    def __init__(self, configs, retrieved_syllabi_chunks):
        self.configs = configs
        self.retrieved_syllabi_chunks = retrieved_syllabi_chunks


    def __call__(self, batch):
        # Construct text
        prompts = [self.create_prompt_search_baseline(item, self.retrieved_syllabi_chunks) for item in batch]

        return {
            "prompts": prompts
            }


    def create_prompt_search_baseline(self, item, retrieved_syllabi_chunks):
        rag_context = retrieved_syllabi_chunks[item["id"]][0] 
        prompt = f"{rag_context}"

        return prompt