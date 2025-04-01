from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import boto3
from io import StringIO

# AWS Configuration
AWS_ACCESS_KEY_ID = 'your_id'  
AWS_SECRET_ACCESS_KEY = 'your_key'  
S3_BUCKET_NAME = 'bucket_name'  

def extract_data(**kwargs):
    df = pd.read_csv('/opt/airflow/data/train.csv')  
    return df.to_json()

#Clean and prepare the data for training
def transform_data(**kwargs):
    ti = kwargs['ti']
    json_data = ti.xcom_pull(task_ids='extract_data')
    df = pd.read_json(json_data)
    
    df['comment_text'] = df['comment_text'].str.lower()
    df['comment_text'] = df['comment_text'].str.replace('[^\w\s]', '')
    
    toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['is_toxic'] = df[toxicity_columns].max(axis=1)
    df = df[['comment_text', 'is_toxic']]
    
    return df.to_json()

#Upload processed data to S3
def load_data(**kwargs):
    ti = kwargs['ti']
    json_data = ti.xcom_pull(task_ids='transform_data')
    df = pd.read_json(json_data)

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
   
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key='processed_data/toxic_comments_processed.csv',
        Body=csv_buffer.getvalue()
    )

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'toxic_comment_etl',
    default_args=default_args,
    description='ETL pipeline for toxic comments',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

extract_task >> transform_task >> load_task