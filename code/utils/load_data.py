import torch
import json

from code.utils.utils import load_df


def load_data(configs):
    data = {}
    for name in ["train", "val", "test"]:
        df = load_df(f"{name}", configs.data_dir)
        json_out = df.to_json(orient="records")
        data[name] = json.loads(json_out)
    
    # Debug with less data
    if(configs.debug):
        data["train"] = data["train"][:2]
        data["val"] = data["val"][:2]
        data["test"] = data["test"][:8]

    print(f"Loaded {len(data['train'])} train samples, {len(data['val'])} val samples, {len(data['test'])} test samples.")
    
    return data["train"], data["val"], data["test"]


def get_data_loaders(train_set, val_set, collate_fn, tokenizer, device, configs, df_retrieved_syllabi_chunks=None):
    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn(tokenizer, device, configs, df_retrieved_syllabi_chunks), 
                                                batch_size=configs.batch_size, num_workers=configs.num_workers, shuffle=True, drop_last=False)                        
    val_loader = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn(tokenizer, device, configs, df_retrieved_syllabi_chunks), 
                                                batch_size=configs.val_batch_size, num_workers=configs.num_workers, shuffle=False, drop_last=False)
    
    return train_loader, val_loader


def get_test_data_loader(test_set, collate_fn, tokenizer, device, configs, df_retrieved_syllabi_chunks=None):
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn(tokenizer, device, configs, df_retrieved_syllabi_chunks), 
                                                batch_size=configs.test_batch_size, num_workers=configs.num_workers, shuffle=False, drop_last=False)
    
    return test_loader


def get_test_data_loader_search_baseline(test_set, collate_fn, configs, df_retrieved_syllabi_chunks):
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn(configs, df_retrieved_syllabi_chunks), 
                                                batch_size=configs.test_batch_size, num_workers=configs.num_workers, shuffle=False, drop_last=False)
    
    return test_loader