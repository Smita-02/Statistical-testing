import os
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fraud_detection_project import logger
from fraud_detection_project.entity.config_entity import DataTransformationConfig

class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def read_data(data_path: str) -> pd.DataFrame:
        """
        It reads a CSV file and returns a Pandas DataFrame

        Args:
          file_path (str): The path to the file you want to read.

        Returns:
          A dataframe
        """
        try:
            logger.info("reading the Dataset")
            return pd.read_csv(data_path)

        except Exception as e:
            raise Exception(e)
        

    def extract_date_features(self, data_path):
        
        try:
            
            self.data: pd.DataFrame = DataTransformation.read_data(data_path)
            logger.info("Deriving the Date features from Transactions Date")
            # Convert the date column to datetime format
            self.data['trans_date_trans_time'] = pd.to_datetime(self.data['trans_date_trans_time'], format='%d-%m-%Y')
            
            # Extract month, year, and quarter
            self.data['trans_month'] = self.data['trans_date_trans_time'].dt.month
            self.data['trans_year'] = self.data['trans_date_trans_time'].dt.year
            self.data['trans_quarter'] = self.data['trans_date_trans_time'].dt.quarter
            
            logger.info("Derived the new Date features from Transactions Date")

            return self.data 
            
        except Exception as e:
            raise Exception("Error:", e)


    def calculate_customer_age(self, data_path):
       
       try:
            
            self.data = self.extract_date_features(data_path)
            
            logger.info("Creating the Customer Age variable using DOB and Transactions Date")
            
            self.data['dob'] = pd.to_datetime(self.data['dob'], format='%d-%m-%Y')
            self.data['Cust_age'] = (self.data['trans_date_trans_time'] - self.data['dob']).dt.days // 365
            
            logger.info("Created the Customer Age variable")

            return self.data

       except Exception as e:
            raise Exception("Error:", e)
                        

    def create_city_population_bins(self, data_path):
        try:
            
            self.data = self.calculate_customer_age(data_path)

            logger.info("Creating the city populations category by binning the city population")

            bins = [0, 5000, 50000, float('inf')]
            labels = ['rural', 'sub-urban', 'urban']
        
            self.data['city_pop_category'] = pd.cut(self.data['city_pop'], bins=bins, labels=labels)

            logger.info("Created the city_pop_category")

            return self.data
        except Exception as e:
            raise Exception("Error:", e)
        

    def calculate_average_amount_by_category(self, data_path):

        try:
            
            self.data = self.create_city_population_bins(data_path)

            logger.info("Creating the avg_amount_by_category variable")

            avg_amount_by_category = self.data.groupby('category')['amt'].mean()
            self.data['avg_amount_by_category'] = self.data['category'].map(avg_amount_by_category)
            self.data = self.data.drop(labels=["trans_date_trans_time", 'dob'], axis=1)
            return self.data

        except Exception as e:
            raise Exception("Error", e)
    


    def label_encoding(self, data_path):
        try:
            
            self.data = self.calculate_average_amount_by_category(data_path)

            logger.info("performing the Label encoding on Category and city_pop_category")

            encoded_data = self.data.copy()
            columns = ['category', 'city_pop_category']
            for col in columns:
                encoder = LabelEncoder()
                encoded_data[col] = encoder.fit_transform(self.data[col])
            return encoded_data
        
        except Exception as e:
            raise Exception("Error", e)
        
    def one_hot_encode_columns(self, data_path):
        try:
           
            self.data = self.label_encoding(data_path)

            logger.info("performing the one hot encoding on gender and trans_year")
            
            cols = ['gender', 'trans_year']
            encoded_data = self.data.copy()
            encoded_data = pd.get_dummies(encoded_data, columns=cols, dtype=int)

            logger.info("Done with feature encoding")
            
            return encoded_data
            
        except Exception as e:
            raise Exception("Error", e)

        
    def fit_transform(self, data_path):
        try:
            
            self.data = self.one_hot_encode_columns(data_path)

            logger.info("performing feature scaling")

            columns_to_scale = ['amt', 'Cust_age', 'city_pop', 'avg_amount_by_category']
            scaler = StandardScaler()
            scaled_data = self.data.copy()
            scaled_data[columns_to_scale] = scaler.fit_transform(self.data[columns_to_scale])

            logger.info("Feature Scaling process is done")
        
            return scaled_data
        
        except Exception as e:
            raise Exception("Error", e)
            

    def final_data(self, data_path):
        try:
            
            self.data = self.one_hot_encode_columns(data_path)

            logger.info("Splitting Dataset into training and test sets Started")
            
            # Split the data into training and test sets. (0.75, 0.25) split.
            train, test = train_test_split(self.data, test_size=0.2, random_state=42)

            train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

            logger.info("Completed the split and stored Splited data into training and test sets")
            logger.info(train.shape)
            logger.info(test.shape)

            print(train.shape)
            print(test.shape)
        except Exception as e:
            raise Exception("Error", e)