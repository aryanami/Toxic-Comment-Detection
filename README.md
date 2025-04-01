# Toxic Comment Detection System

## Objective
This project aims to automatically detect toxic content in online comments using machine learning. The system classifies text into toxic/non-toxic categories to help platforms moderate content and maintain healthy online communities.

## Key Features
- **Real-time toxicity detection** - Analyze text for harmful content instantly
- **Multi-label classification** - Detects various toxicity types (threats, insults, etc.)
- **Scalable API** - Deployed on AWS for high availability
- **Simple interface** - Easy-to-use web interface for testing
- **Data pipeline** - Automated ETL process for model retraining

## Tech Stack
### Core Components
- **Machine Learning**: DistilBERT (distilled version of BERT)
- **Backend**: Flask API with Python 
- **Data Processing**: Pandas, Numpy
- **Model Serving**: Docker, AWS ECR/ECS

### Infrastructure
- **Orchestration**: Apache Airflow
- **Cloud Services**: AWS (EC2, S3, Lambda, API Gateway)
- **Containerization**: Docker
- **Version Control**: Git, GitHub

## Implementation

### Data Pipeline
1. **Extract**: Pull raw comment data from Kaggle
2. **Transform**: Clean text and combine toxicity labels
3. **Load**: Store processed data in S3 bucket

### Model Development
1. Fine-tuned DistilBERT on toxic comment dataset
2. Achieved ~95% accuracy on validation set
3. Optimized model for production deployment

### Deployment Architecture
1. Docker container with Flask API endpoint
2. Hosted on AWS ECS with API Gateway frontend
3. Auto-scaling configured for high traffic loads

## Conclusion
This project demonstrates a complete ML pipeline from data processing to production deployment. The DistilBERT model provides near state-of-the-art accuracy while being more efficient than full BERT. The AWS infrastructure ensures scalability and reliability for real-world usage.

Future improvements could include:
- Expanding to more toxicity categories
- Adding multilingual support
- Implementing active learning for continuous improvement
