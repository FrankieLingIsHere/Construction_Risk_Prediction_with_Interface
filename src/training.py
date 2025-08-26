import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import random

from huggingface_hub import login
login(token="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")
from huggingface_hub import notebook_login
from huggingface_hub import create_repo, upload_folder

# Configuration
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "your_jsonl_file.jsonl"  # Updated path
OUTPUT_DIR = "./your_target_output_directory"
MAX_LENGTH = 512  # Keep this - it's sufficient

def format_input(scenario_text):
    """Format input the same way as in your cascade predictor"""
    scenario = scenario_text.strip()
    return f"Based on the situation, predict potential hazards and injuries. {scenario}\nOutput:\n"

def load_and_prepare_dataset(file_path):
    """Load and prepare dataset without train/test split"""
    print(f"Loading dataset from: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Dataset file not found: {file_path}")
        print("Available .jsonl files:")
        for file in os.listdir('.'):
            if file.endswith('.jsonl'):
                print(f"   - {file}")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = []
    problematic_lines = []
    line_num = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_num += 1
            if not line.strip():
                continue

            try:
                item = json.loads(line.strip())

                # Validate required fields
                if 'input' not in item or 'output' not in item:
                    problematic_lines.append(f"Line {line_num}: Missing 'input' or 'output' field")
                    continue

                output = item['output']
                required_fields = ["Hazards", "Cause of Accident", "Degree of Injury"]
                missing_fields = [field for field in required_fields if field not in output]

                if missing_fields:
                    problematic_lines.append(f"Line {line_num}: Missing output fields: {missing_fields}")
                    continue

                # Format training example
                prompt = format_input(item['input'])
                output_json = json.dumps(output, separators=(',', ':'))
                training_text = prompt + output_json

                data.append({
                    'text': training_text,
                    'input_length': len(prompt),
                    'expected_output': output_json,
                    'line_number': line_num
                })

            except json.JSONDecodeError as e:
                problematic_lines.append(f"Line {line_num}: JSON decode error - {str(e)}")
                continue

    print(f"✅ Loaded {len(data)} valid training examples")

    if problematic_lines:
        print(f"⚠️  Found {len(problematic_lines)} problematic lines:")
        for issue in problematic_lines[:10]:  # Show first 10
            print(f"   {issue}")
        if len(problematic_lines) > 10:
            print(f"   ... and {len(problematic_lines) - 10} more issues")

    if len(data) == 0:
        raise ValueError("No valid training examples found!")

    # Shuffle data for better training
    random.shuffle(data)
    print(f"📊 Dataset prepared with {len(data)} examples (shuffled)")

    return data

class DataCollator:
    """Enhanced data collator with better error handling"""
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item['text'] for item in batch]

        # Tokenize all texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (copy of input_ids)
        labels = encoded['input_ids'].clone()

        # Mask the input part (only train on output)
        for i, item in enumerate(batch):
            try:
                # Tokenize just the input part to find where to start learning
                input_text = item['text'][:item['input_length']]
                input_tokens = self.tokenizer(input_text, add_special_tokens=False)['input_ids']
                input_length = len(input_tokens)

                # Mask input tokens (set to -100 so they're ignored in loss)
                if input_length < labels.shape[1]:
                    labels[i, :input_length] = -100

            except Exception as e:
                print(f"Warning: Error processing batch item {i}: {e}")
                # If there's an error, don't mask anything (less efficient but safe)
                pass

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        encoded['labels'] = labels
        return encoded

def setup_model():
    """Setup model with enhanced LoRA configuration"""
    print("🔄 Loading base model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    print(f"   Model vocab size: {len(tokenizer)}")

    # Enhanced LoRA configuration for better JSON structure learning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Increased from 8 for better capacity
        lora_alpha=32,  # Increased from 16
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # More modules
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.train()

    print("✅ Model setup complete!")
    print(f"   Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    print(f"   Total parameters: {model.num_parameters():,}")

    return model, tokenizer

def test_model_after_training(model, tokenizer):
    """Quick test to verify the model produces complete JSON - FIXED DEVICE ISSUE"""
    print("\n🧪 Testing trained model...")

    test_scenario = "A worker fell from a ladder while working at height without safety harness."
    prompt = format_input(test_scenario)

    # 🔧 FIX: Get the device that the model is on
    device = next(model.parameters()).device
    print(f"   Model is on device: {device}")

    # 🔧 FIX: Put inputs on the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to model device

    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📝 Test Input: {test_scenario}")
    print(f"📤 Model Output:\n{result}")

    # Check if output contains all required fields
    try:
        json_start = result.find("{")
        json_end = result.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_output = result[json_start:json_end]
            parsed = json.loads(json_output)

            required_fields = ["Hazards", "Cause of Accident", "Degree of Injury"]
            missing_fields = [field for field in required_fields if field not in parsed]

            if not missing_fields:
                print("✅ SUCCESS: Model produces complete JSON with all required fields!")
                return True
            else:
                print(f"⚠️  WARNING: Missing fields in output: {missing_fields}")
                return False
        else:
            print("⚠️  WARNING: No valid JSON structure found in output")
            return False

    except Exception as e:
        print(f"⚠️  WARNING: Could not parse JSON output: {e}")
        return False

    finally:
        model.train()  # Return to training mode

def train():
    """Main training function without train/test split"""
    print("🚀 Starting Model 3 Training (Complete Dataset)")
    print("=" * 60)

    try:
        # Load dataset (no splitting)
        data = load_and_prepare_dataset(DATASET_PATH)
        dataset = Dataset.from_list(data)

        # Setup model and tokenizer
        model, tokenizer = setup_model()

        # Create data collator
        data_collator = DataCollator(tokenizer, max_length=MAX_LENGTH)

        # Training arguments optimized for complete dataset training
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=8,  # More epochs since we're not splitting data
            per_device_train_batch_size=1,  # Smaller batch for stability
            gradient_accumulation_steps=8,  # Effective batch size = 8
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_steps=50,  # Reduced since no validation
            logging_steps=25,
            save_steps=100,  # Save more frequently
            save_strategy="steps",
            save_total_limit=3,
            weight_decay=0.01,
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            logging_first_step=True,
            report_to=None,
            fp16=False,
            # No evaluation since we're not splitting data
            eval_strategy="no",
            load_best_model_at_end=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,  # Only training dataset
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        print(f"📊 Training Configuration:")
        print(f"   Dataset: {len(dataset)} examples")
        print(f"   Epochs: {training_args.num_train_epochs}")
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"   Max length: {MAX_LENGTH}")

        # Start training
        print("\n🎯 Starting training...")
        trainer.train()

        print("\n💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)

        # 🔧 FIX: Test model after saving (this ensures proper device handling)
        print("\n🧪 Loading and testing the saved model...")

        # Load the saved model for testing (this handles device placement correctly)
        from peft import PeftModel
        base_model_test = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        test_model = PeftModel.from_pretrained(base_model_test, OUTPUT_DIR)

        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_model = test_model.to(device)

        # Test the loaded model
        test_success = test_model_after_training(test_model, tokenizer)

        print(f"✅ Training completed successfully!")
        print(f"📁 Model saved to: {OUTPUT_DIR}")

        if test_success:
            print("🎉 Model 3 is now ready and produces complete JSON!")
        else:
            print("⚠️  Model may need additional training - JSON output incomplete")

        return True

    except Exception as e:
        import traceback
        print(f"\n❌ Training failed with error: {str(e)}")
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Run training
    success = train()

    if success:
        print("\n" + "="*60)
        print("🎯 NEXT STEPS:")
        print("1. Update your cascade predictor MODEL_PATHS to use:")
        print(f"   './Construction_Risk_TinyLlama_M3_Fixed'")
        print("2. Test the cascade predictor with the fire scenario")
        print("3. Model 3 should now produce complete JSON with all fields!")
        print("="*60)

        #Upload to huggingface as adapter model
        notebook_login()

        # Replace these with your info
        model_name = "your_model_name"
        username = "your_username"  # Your HF username or org name
        local_model_path = "./your_output_model_directory"

        # Full repo ID
        repo_id = f"{username}/{model_name}"

        # ✅ Step 1: Create the repo on HuggingFace
        create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

        # ✅ Step 2: Upload model files to the repo
        upload_folder(
            repo_id=repo_id,
            folder_path=local_model_path,
            path_in_repo=".",  # Root of the repo
            repo_type="model",
            commit_message="Upload merged fine-tuned risk prediction model",
        )
    else:
        print("\n❌ Training failed. Please check the error messages above.")