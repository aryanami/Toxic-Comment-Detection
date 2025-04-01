import json
import boto3
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import io

# Configuration
CONFIG = {
    "bucket_name": "bucket_name",
    "model_path": "models/toxic_distilbert_model",
    "threshold": 0.5  
}

s3 = boto3.client('s3')

#Load DistilBERT model from S3
def load_model():
    try:
        config_obj = s3.get_object(
            Bucket=CONFIG["bucket_name"],
            Key=f"{CONFIG['model_path']}/config.json"
        )
        model_obj = s3.get_object(
            Bucket=CONFIG["bucket_name"],
            Key=f"{CONFIG['model_path']}/pytorch_model.bin"
        )
        
        config_file = io.StringIO(config_obj['Body'].read().decode('utf-8'))
        model_file = io.BytesIO(model_obj['Body'].read())
        
        tokenizer = DistilBertTokenizer.from_pretrained(CONFIG["model_path"])
        model = DistilBertForSequenceClassification.from_pretrained(
            CONFIG["model_path"],
            state_dict=torch.load(model_file, map_location=torch.device('cpu')),
            config=json.load(config_file)
        )
        
        return tokenizer, model
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model from S3: {str(e)}")

# Load model when Lambda initializes (cold start)
try:
    tokenizer, model = load_model()
    model.eval()  
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading error: {str(e)}")
    raise

def lambda_handler(event, context):
    """Lambda function handler"""
    try:
        # Get text from API Gateway
        text = event.get('queryStringParameters', {}).get('text', '')
        
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No text provided'})
            }
     
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        toxic_prob = probabilities[0][1].item()
        is_toxic = toxic_prob > CONFIG["threshold"]
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'is_toxic': is_toxic,
                'probability': toxic_prob,
                'text': text
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error'
            })
        }