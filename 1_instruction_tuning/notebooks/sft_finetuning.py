from huggingface_hub import login
login()
# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "SmolLM2-FT-MyDataset"
finetune_tags = ["smol-course", "module_1"]

# Let's test the base model before training
prompt = "How are you ?"

# Format with template
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print("Before training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Load a sample dataset


# TODO: define your dataset and config using the path and name parameters
ds_raw = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")



def process_dataset(sample):
    tokenized_messages = tokenizer.apply_chat_template(sample["messages"], add_generation_prompt=True)
    sample["messages"] = tokenized_messages
    return sample


ds = ds_raw.map(process_dataset)


# TODO: 🦁 If your dataset is not in a format that TRL can convert to the chat template, you will need to process it. Refer to the [module](../chat_templates.md)

# Configure the SFTTrainer
sft_config = SFTConfig(
    output_dir="./sft_output_scar1",
    max_steps=1000,  # Adjust based on dataset size and desired training duration
    per_device_train_batch_size=4,  # Set according to your GPU memory capacity
    learning_rate=5e-5,  # Common starting point for fine-tuning
    logging_steps=10,  # Frequency of logging training metrics
    save_steps=100,  # Frequency of saving model checkpoints
    evaluation_strategy="steps",  # Evaluate the model at regular intervals
    eval_steps=50,  # Frequency of evaluation
    use_mps_device=(
        True if device == "mps" else False
    ),  # Use MPS for mixed precision training
    hub_model_id=finetune_name,  # Set a unique name for your model
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds_raw["train"],
    tokenizer=tokenizer,
    eval_dataset=ds_raw["test"],
)

# TODO: 🦁 🐕 align the SFTTrainer params with your chosen dataset. For example, if you are using the `bigcode/the-stack-smol` dataset, you will need to choose the `content` column`

# Train the model
trainer.train()

# Save the model
trainer.save_model(f"./{finetune_name}")
trainer.push_to_hub(tags=finetune_tags)

