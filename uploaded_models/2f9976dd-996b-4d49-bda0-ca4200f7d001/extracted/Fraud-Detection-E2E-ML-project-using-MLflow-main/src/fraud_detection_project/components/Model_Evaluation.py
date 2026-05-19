import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import joblib
from fraud_detection_project.entity.config_entity import ModelEvaluationConfig
from fraud_detection_project.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def evaluate_model(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        return accuracy, precision, recall
        

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            (accuracy, precision, recall) = self.evaluate_model(test_y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGBClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")

