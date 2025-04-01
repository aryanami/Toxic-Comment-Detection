import os
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import boto3
from io import StringIO
from tqdm import tqdm
import numpy as np

# Configuration
CONFIG = {
    "aws_access_key_id": "your_id",  
    "aws_secret_access_key": "your_key", 
    "s3_bucket": "bucket_name",
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 2,
    "learning_rate": 2e-5,
    "sample_fraction": 0.1,
    "grad_clip": 1.0,  
    "seed": 42
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

#Load a fraction of the processed data from S3
def load_sample_data():
    s3 = boto3.client(
        's3',
        aws_access_key_id=CONFIG["aws_access_key_id"],
        aws_secret_access_key=CONFIG["aws_secret_access_key"]
    )
    
    obj = s3.get_object(
        Bucket=CONFIG["s3_bucket"],
        Key='processed_data/toxic_comments_processed.csv'
    )
    
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    return df.sample(frac=CONFIG["sample_fraction"], random_state=CONFIG["seed"])

#Upload model files to S3
def upload_model_to_s3(local_path, s3_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=CONFIG["aws_access_key_id"],
        aws_secret_access_key=CONFIG["aws_secret_access_key"]
    )
    
    # Upload each file in the model directory
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = os.path.join(
                s3_path,
                os.path.relpath(local_file_path, local_path)
            ).replace('\\', '/')  
            
            s3.upload_file(
                local_file_path,
                CONFIG["s3_bucket"],
                s3_file_path
            )
            print(f"Uploaded {local_file_path} to s3://{CONFIG['s3_bucket']}/{s3_file_path}")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    print("Loading sample data...")
    df = load_sample_data()
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=CONFIG["seed"]
    )
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG["model_name"])
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=2
    ).to(device)
    
    # Create datasets
    train_dataset = ToxicCommentDataset(
        train_df.comment_text.values,
        train_df.is_toxic.values,
        tokenizer,
        CONFIG["max_length"]
    )
    
    val_dataset = ToxicCommentDataset(
        val_df.comment_text.values,
        val_df.is_toxic.values,
        tokenizer,
        CONFIG["max_length"]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"]
    )
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Learning rate scheduler
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    print(f"\nStarting training with:")
    print(f"- Samples: {len(train_df)} train, {len(val_df)} validation")
    print(f"- Batch size: {CONFIG['batch_size']}")
    print(f"- Epochs: {CONFIG['epochs']}\n")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions.double() / len(val_dataset)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"- Training Loss: {avg_train_loss:.4f}")
        print(f"- Validation Loss: {avg_val_loss:.4f}")
        print(f"- Validation Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "./toxic_distilbert_model"
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"\nSaved best model to {save_path} with validation loss: {avg_val_loss:.4f}")
            upload_model_to_s3(save_path, "models/toxic_distilbert_model")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    train()
