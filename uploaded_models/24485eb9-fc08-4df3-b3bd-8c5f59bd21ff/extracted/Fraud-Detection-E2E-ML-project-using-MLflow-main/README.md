# End to End Fraud Detection ML project using MLflow & AWS

<img src="https://github.com/pavannagula/Fraud-Detection-E2E-ML-project-using-MLflow/blob/main/static/assets/image.jpg" width="1100" height="500" />

Photo by [iStock](https://www.istockphoto.com/photo/technology-photos-gm915806412-252027852?phrase=credit+card+fraud)

## Table of Contents
- [Introduction](#introduction)
    - [Project Overview](#project-overview)
    - [Languages & Tools](#languages-and-tools)
- [Business Problem](#business-problem)
- [Dataset Source](#dataset-source)
- [Project Workflow](#project-workflow)
  - [Data Ingestion](#data-ingestion)
  - [Data Validation](#data-validation)
  - [Data Transformation](#data-transformation)
  - [Model Building](#model-building)
  - [Model Evaluation](#model-evaluation)
  - [Model Deployment](#model-deployment)
- [References](#references)
- [License](#license)


## Introduction
> Fraud detection plays a pivotal role in maintaining the integrity and security of financial systems. The rise of digital transactions has necessitated robust techniques to identify and prevent fraudulent activities in credit card transactions. This project delves into the realm of fraud detection, employing advanced machine learning algorithms to detect potentially fraudulent credit card transactions.

### Project Overview
> This repository showcases an end-to-end workflow for building a fraud detection model using state-of-the-art techniques. The project encompasses every stage of the data science lifecycle, from data ingestion to model deployment.

### Languages and Tools
 > - **Python**: The core programming language utilized for data preprocessing, model building, building pipelines and deployment.
 > - **AWS Services**: Amazon Web Services (AWS) services have been leveraged extensively. AWS S3 is utilized for data ingestion, AWS EC2 instances host the model, AWS ECR stores Docker images, and more.
 > - **Docker**: Docker is used to containerize the model and create portable Docker images.
 > - **Flask**: The Flask framework is employed to create a web application that interacts with the deployed model, providing a user-friendly interface.
 > - **MLflow**: MLflow is utilized for experiment tracking and management, ensuring efficient experimentation and reproducibility.
 > - **DagsHub**: DagsHub is also used to serve as a remote server for this Git repositories, facilitating collaboration and version control in the project.
 > - **Pandas, NumPy, scikit-learn**: Widely-used Python libraries for data manipulation, analysis, and machine learning.


## Business Problem:

> In today's digital age, where electronic transactions have become an integral part of our lives, the security and integrity of financial systems are of paramount importance. The rapid growth of online financial activities has led to a rise in fraudulent activities, demanding robust strategies to safeguard the interests of both financial institutions and customers.So, the main aim of this project is to create a fraud detection model capable of distinguishing between legitimate and fraudulent credit card transactions

## Dataset Source
> The dataset used in this project is sourced from Kaggle. It comprises a simulated credit card transaction history covering the period from January 1st, 2019 to December 31st, 2020. This dataset includes a diverse range of transactions, incorporating both legitimate and fraudulent activities.

**Dataset link:** [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)



## Project Workflow

### Data Ingestion
> In the data ingestion stage, I seamlessly integrated AWS services into my workflow. The data was uploaded to an S3 bucket, and by leveraging the AWS CLI, I efficiently extracted the data from the S3 bucket. This data was then stored in the designated data ingestion artifacts directory. This strategic approach ensures my project's data remains organized and readily accessible for subsequent stages of the project.

Data Ingestion Pipeline - `Data_Ingestion.py` -> Connected to AWS S3 bucket and retrived the dataset

### Data Validation
> In the Data Validation stage, I initiated by reading the data from the data ingestion artifacts folder. To ensure data integrity and adherence to the project's requirements, I subjected the data to rigorous validation against a predefined schema of column names. Furthermore, as part of this validation process, I incorporated a status file creation mechanism. This status file serves as a definitive record, indicating whether the validation was successful or encountered issues.

Data Validation Pipeline - `Data_Validation.py` -> Data Validation steps which involves validation of Pre-defined schema and returns validation status.

### Data Transformation
> During the transformation stage, I used the "chain of reaction" method to build the whole data transformation pipeline which included feature engineering, data transformation and feature scaling. By interlinking these components in a sequential manner, I ensured that the data flowed seamlessly from one step to the next, resulting in a streamlined and automated transformation process

Data Transformation Pipeline - `Data_Transformation.py` -> which involved Chaining Method to build the whole transformation pipeline.

### Model Building
> In the model-building stage, I split the dataset into training and testing subsets and trained an XGBoost classifier using the defined parameters from the params.yaml file. The trained model was then stored as "model.joblib" in the model training artifacts folder, ready for deployment. This approach ensures an efficient and reproducible process for creating and archiving the model for future use

Model Building Pipeline - `Model_Trainer.py` -> Involved train test split and XGBoost Model training

### Model Evaluation
> Moving on to the model evaluation phase, I rigorously assessed the model's performance using the test dataset. Employing diverse hyperparameters, I fine-tuned the model to achieve optimal results. MLflow played a pivotal role in this process by enabling experiment tracking, allowing me to efficiently compare various model iterations. The best-performing model was then stored in the model evaluation artifacts folder for future reference. Additionally, I utilized Dagshub as a remote server to facilitate version control, ensuring the seamless management of model versions and improvements throughout this phase.

Model Evaluation Pipeline - `Model_Trainer.py` -> Connected to MLflow to track the Model performance by hosting the app in Dagshub remote server. 

## Model Experiments Comparision in MLflow

![image](https://github.com/pavannagula/Fraud-Detection-E2E-ML-project-using-MLflow/assets/39379433/1a77085d-29a2-46b4-8f68-1a2e46bfdde8)


### Mlflow
[Mlflow](https://mlflow.org/docs/latest/index.html)

### Dagshub
[Dagshub](https://dagshub.com/)

### Model Deployment
> Transitioning to the model deployment stage, I used Flask to construct a web application that interacted with users on the front end. This application incorporated the prediction pipeline which takes users input information and returns real-time predictions. The resulting predictions were then presented on the results homepage. Upon successful validation and functionality, I compiled a Docker image encapsulating the entire application. Subsequently, I established an Amazon EC2 instance and installed Docker within it. Employing GitHub Actions, I streamlined the integration of the repository with the EC2 instance, ensuring a continuous and automated deployment process. This comprehensive approach facilitated the successful deployment and accessibility of the predictive model.

    # Steps for AWS CI/CD deployment in EC2 using Github Actions
        1. Created IAM  User with full Access to ECR & EC2 policies
        2. Built Docker Image of Source Code and Pushed it into ECR
        3. Pulled Docker Image in EC2 from ECR and launched it. 
        4. Setting EC2 instance as Self hosted Runner to connect it with Github by using Github Actions
        5. Created Action variables to make a connection with EC2 instance
        6. Post commit the CI & CD Deployment starts and once its done, Added the port to EC2 instance.

## References
- https://github.com/entbappy
- https://www.youtube.com/@krishnaik06

## License
[MIT Licence](https://github.com/pavannagula/Fraud-Detection-E2E-ML-project-using-MLflow/blob/main/LICENSE)
