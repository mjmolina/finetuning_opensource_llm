import bitsandbytes as bnb
import torch
from datasets import load_dataset
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, pipeline)
from trl import SFTTrainer


def load_model(model_name):
    """
    Load the quantization configuration and the model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # Use 4-bit Normal Float for storing the base model weights in GPU memory
        bnb_4bit_compute_dtype=torch.float16  # De-quantize the weights to 16-bit float before the forward/backward pass
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config)
    model.config.use_cache = False
    print(model)
    return model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def user_prompt(human_prompt):
    prompt_template = (
        f"### HUMAN:\n{human_prompt}\n\n### RESPONSE:\n"  # This depends on the dataset format.
    )
    return prompt_template


def create_lora_config(lora_alpha, lora_dropout, lora_r):
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, # Drop out ratio for the layers in LoRA adaptors.
        r=lora_r, # This is the rank of the decomposed matrices to be learned during fine-tuning.
        bias="none", # Bias parameters to train.
        task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], #This depends on the model and fine-tune the linear layers in the model.
    )
    return peft_config


def get_training_arguments():
    return TrainingArguments(
        output_dir="Trainer_output",
        # For the following arguments, refer to https://huggingface.co/docs/transformers/main_classes/trainer
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        optim="paged_adamw_8bit",
        fp16=True,
        max_grad_norm=0.3,
        max_steps=500,
        warmup_ratio=0.03,
        group_by_length=True,
        logging_steps =100,
        save_steps =50,
        lr_scheduler_type="constant",
    )


def get_trainer(model, peft_config, training_arguments, dataset_train):
    return SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        dataset_text_field="text",
        peft_config=peft_config,
        args=training_arguments,
        max_seq_length=512,
    )


def test_model(model, tokenizer):
    pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=150,
        repetition_penalty=1,
        #top_p=0.95,
    )

    result = pipeline(user_prompt("Eres una IA con conocimientos generales muy avanzados. Que es el cisne negro?"))
    print(result[0]["generated_text"])


if __name__ == "__main__":
    dataset = load_dataset("hlhdatscience/guanaco-spanish-dataset", split="train")
    model_name = "ericzzz/falcon-rw-1b-instruct-openorca"
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    model2 = prepare_model_for_kbit_training(model)
    peft_config = create_lora_config(lora_alpha = 16,
                                     lora_dropout = 0.05,
                                     lora_r = 8)
    peft_model = get_peft_model(model2,peft_config)
    print(peft_model)
    print(peft_model.print_trainable_parameters())

    # Test initial model
    test_model(model, tokenizer)

    
    # Set up the model for quantization-aware training e.g. casting layers, parameter freezing, etc.
    training_arguments = get_training_arguments()
    trainer = get_trainer(model, peft_config, training_arguments, dataset)
    trainer.train()
    trainer.save_model("Finetuned_adapter")

    # Trained Model + peft
    adapter_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16), "Finetuned_adapter")
    merged_model = adapter_model.merge_and_unload()
    merged_model.save_pretrained("Merged_model")
    tokenizer.save_pretrained("Merged_model")

    # Test merged model
    test_model(merged_model, tokenizer)
