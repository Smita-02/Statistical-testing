from fraud_detection_project.config.configuration import *
from fraud_detection_project.components.Data_Transformation import DataTransformation
from fraud_detection_project import logger


STAGE_NAME = "Data Transformation"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.final_data('artifacts/data_ingestion/fraud_data.csv')

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)

    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e