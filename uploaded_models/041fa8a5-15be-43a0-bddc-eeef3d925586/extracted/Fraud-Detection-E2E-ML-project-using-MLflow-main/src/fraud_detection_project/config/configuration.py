from fraud_detection_project.constants import *
from fraud_detection_project.utils.common import read_yaml, create_directories
from fraud_detection_project.entity.config_entity import (DataIngestionConfig, 
                                                         DataValidationConfig,
                                                         DataTransformationConfig,
                                                         ModelTrainerConfig,
                                                         ModelEvaluationConfig)

class ConfigurationManager:
    # First setting up the environment by defining the file paths of configuration file, params file and schema file.  
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        # Creating the root directory in the artifacts folder
        create_directories([self.config.artifacts_root])


    # Data Ingestion step
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            # We are accessing the file paths that are defined in the configuration yaml file
            root_dir=config.root_dir,
            s3_bucket=config.s3_bucket,
            s3_object_key=config.s3_object_key,
            local_data_file=config.local_data_file,
        )

        return data_ingestion_config
    
    # Data Validation step
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir = config.data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    # Data Transformation stage
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    
    # Model Training Stage
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.XGBClassifier
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            n_estimators = params.n_estimators,
            max_depth = params.max_depth,
            learning_rate = params.learning_rate,
            subsample = params.subsample,
            colsample_bytree = params.colsample_bytree,
            random_state = params.random_state,
            target_column = schema.name
            
        )

        return model_trainer_config
    
    # Model Evaluatoin Stage:
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.XGBClassifier
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri="https://dagshub.com/pavannagula/Fraud-Detection-E2E-ML-project-using-MLflow.mlflow",
           
        )

        return model_evaluation_config