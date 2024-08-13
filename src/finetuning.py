import openai
import time

# Replace with your OpenAI API key
openai.api_key = "your-openai-api-key"

# Replace with the path to dataset file
dataset_path = "your_dataset.jsonl"

# Upload the dataset
def upload_dataset(file_path):
    print("Uploading dataset...")
    upload_response = openai.File.create(
        file=open(file_path, "rb"),
        purpose='fine-tune'
    )
    print("Uploaded file ID:", upload_response["id"])
    return upload_response["id"]

# Fine-tune the model
def fine_tune_model(file_id):
    print("Starting fine-tuning...")
    fine_tune_params = {
        "training_file": file_id,
        "model": "gpt-3.5-turbo",  # Base model
        "n_epochs": 3,             # Number of epochs for fine-tuning
        "batch_size": 2,           # Batch size
        "learning_rate_multiplier": 2, # Learning rate multiplier
        "prompt_loss_weight": 0.01 # How much weight to give to the prompt in calculating loss
    }
    response = openai.FineTune.create(**fine_tune_params)
    print("Fine-tuning job created with ID:", response["id"])
    return response["id"]

# Monitor the fine-tuning job
def monitor_fine_tuning(job_id):
    print("Monitoring fine-tuning job...")
    while True:
        status = openai.FineTune.retrieve(id=job_id)
        print("Job status:", status["status"])
        if status["status"] in ["succeeded", "failed"]:
            break
        time.sleep(60)  # Wait for a minute before checking again
    print("Fine-tuning job finished with status:", status["status"])
    return status["fine_tuned_model"]

# Use the fine-tuned model
def use_fine_tuned_model(model_id, prompt):
    print("Using fine-tuned model...")
    response = openai.Completion.create(
        model=model_id,
        prompt=prompt
    )
    return response["choices"][0]["text"].strip()

if __name__ == "__main__":
    # Upload the dataset
    file_id = upload_dataset(dataset_path)

    # Fine-tune the model
    job_id = fine_tune_model(file_id)

    # Monitor the fine-tuning process
    fine_tuned_model_id = monitor_fine_tuning(job_id)

    # Test the fine-tuned model
    test_prompt = "How can you help me?"
    result = use_fine_tuned_model(fine_tuned_model_id, test_prompt)
    print("Model output:", result)
