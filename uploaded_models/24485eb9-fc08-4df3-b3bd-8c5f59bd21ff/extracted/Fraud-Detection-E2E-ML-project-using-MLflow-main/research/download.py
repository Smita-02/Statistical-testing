# Download csv file from S3 bucket
# Code snippet from c:\Users\pavan\Desktop\Fraud-Detection-E2E-ML-project-using-MLflow\src\fraud_detection_project\utils\common.py
import boto3
import botocore   
#from pathlib import path

def download_file(bucket_name, object_name, file_name):
    """
    Download a file from an S3 Bucket
    :param str bucket: Name of the s3 bucket.
    :param str objecyName: name of the key in that bucket to get data for (e.g., 'data/train')
    :return: None if successful else raise exception
    Example usage:
    download_file('my-s3-bucket', 'data/test/', '/tmp/')
    # downloads all files under /data folder into tmp directory
    """

    try:
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, object_name, file_name)
        print("File downloaded successfully")
        return True
    except Exception as e:
        print(e)
        return False
    
    
 