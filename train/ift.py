from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

device = "cuda"
model_name = "jekunz/smollm-135m-fineweb-swedish-from-scratch"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

finetune_name = "jekunz/smollm-135m-fineweb-swedish-from-scratch-smol-smoltalk"

ds = load_dataset(path="liu-nlp/smol-smoltalk-swedish")

sft_config = SFTConfig(
    output_dir="./smollm-135m-fineweb-swedish-from-scratch-smol-smoltalk",
    per_device_train_batch_size=1, 
    learning_rate=5e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs = 1,
    logging_steps=50, 
    save_steps=1000, 
    eval_strategy="no",  
    push_to_hub=True,
    hub_model_id=finetune_name,  
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    eval_dataset=None,
)

trainer.train()