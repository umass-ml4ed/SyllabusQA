import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


class LanguageModel(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self._bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            )   
        self._peft_config = LoraConfig(
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            r=configs.lora_r,
            bias="none",
            task_type="CAUSAL_LM", 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"], # All linear layers
            inference_mode=False
            )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            configs.model_name,
            quantization_config=self._bnb_config,
            device_map="auto"
            )
        # https://github.com/huggingface/transformers/pull/24906
        self._hf_model.config.pretraining_tp = 1 
        self._hf_model = prepare_model_for_kbit_training(self._hf_model)
        self.model = get_peft_model(self._hf_model, self._peft_config)


    def forward(self, **kwargs):
        outputs = self.model(**kwargs)

        return outputs.loss


class Trainer(): 
    def __init__(self, configs, device, num_training_steps):
        self.configs = configs
        self.model = LanguageModel(configs, device).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(configs.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.optimizer = AdamW(self.model.parameters(), lr=configs.lr)
        if( configs.use_lr_scheduler ):
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, configs.num_warmup_steps, num_training_steps)

        
    def zero_grad(self):
        self.optimizer.zero_grad()


    def grad_step(self):
        # Gradient clipping
        if( self.configs.use_grad_clip ):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.grad_clip)
        self.optimizer.step()
        if( self.configs.use_lr_scheduler ):
            self.lr_scheduler.step()


    def train_step(self, batch):
        self.zero_grad()
        loss = self.model(**batch)
        loss.backward()
        self.grad_step()

        return {
            'loss': loss.detach().cpu(),
            "lr": self.lr_scheduler.get_last_lr()[0]
            }


    def val_step(self, batch):
        with torch.no_grad():
            loss = self.model(**batch)

        return {
            "loss": loss.detach().cpu()
            }


    def set_train_mode(self):
        # Recursively sets children to train mode, therefore model.model.train() is also set
        self.model.train()


    def set_eval_mode(self):
        # Recursively sets children to eval mode, therefore model.model.eval() is also set
        self.model.eval()