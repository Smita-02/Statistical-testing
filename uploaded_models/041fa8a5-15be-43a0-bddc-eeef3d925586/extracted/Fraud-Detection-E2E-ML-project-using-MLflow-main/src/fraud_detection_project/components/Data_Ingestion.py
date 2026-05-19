import os
import urllib.request as request
from fraud_detection_project import logger
from fraud_detection_project.utils.common import get_size
from pathlib import Path
from fraud_detection_project.entity.config_entity import (DataIngestionConfig)
import os
import boto3
import yaml
import logging


class DataIngestionConfig:
    def __init__(self, root_dir, s3_bucket, s3_object_key, local_data_file):
        self.root_dir = root_dir
        self.s3_bucket = s3_bucket
        self.s3_object_key = s3_object_key
        self.local_data_file = local_data_file

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.root_dir):
            os.makedirs(self.config.root_dir)

        local_file_path = os.path.join(self.config.root_dir, self.config.local_data_file)

        # Create an S3 client
        s3 = boto3.client('s3')

        # Download the file from S3 if it doesn't exist locally
        if not os.path.exists(local_file_path):
            try:
                s3.download_file(self.config.s3_bucket, self.config.s3_object_key, local_file_path)
                logger.info(f"File downloaded successfully as {local_file_path}")
            except Exception as e:
                logger.error(e)
        else:
            logger.info(f"File already exists: {local_file_path}")

