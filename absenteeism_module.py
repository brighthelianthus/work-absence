
# coding: utf-8

# In[1]:


# libraries

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# the CustomScaler class made taking reference of StandardScaler class
#Standardize features by removing the mean and scaling to unit variance

       # The standard score of a sample `x` is calculated as:

       # z = (x - u) / s , u=mean, s= standard deviation


class CustomScaler(BaseEstimator,TransformerMixin):

    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# create the special class that we are going to use from here on to predict new data
class absenteeism_model():

        def __init__(self, model_file, scaler_file ):
            # to read the 'linear_reg_model' and 'absenteeism_scaler_object' files using pickle
                self.reg = pickle.load( open('linear_reg_model','rb'))
                self.scaler = pickle.load(open('absenteeism_scaler_object', 'rb'))
                self.data = None

        # take a data file (*.csv) and preprocess it
        def load_and_clean_data(self, data_file):

            df = pd.read_csv(data_file,delimiter=',')

            self.df_with_predictions = df.copy()

            df = df.drop(['ID'], axis = 1)
            # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
            df['Absenteeism Time in Hours'] = 'NaN'

            # create a separate dataframe, containing dummy values for all available reasons
            reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)

            # split reason_columns into 4 types
            reason_1 = reason_columns.loc[:,1:15].max(axis=1)
            reason_2 = reason_columns.loc[:,15:18].max(axis=1)
            reason_3 = reason_columns.loc[:,18:22].max(axis=1)
            reason_4 = reason_columns.loc[:,22:].max(axis=1)

            # to avoid multi-collinearity, dropping the 'Reason for Absence' column from df
            df = df.drop(['Reason for Absence'], axis = 1)

            # concatenate df and the 4 reason types for absence
            df = pd.concat([ reason_1, reason_2, reason_3, reason_4, df ], axis = 1)

            # re-name the columns in df
            column_names = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense',
                            'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                    'Children', 'Pet', 'Absenteeism Time in Hours']
            df.columns = column_names

            # convert 'Date' column into datetime datatype from str
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

            # create a list with month values serially retrieved from the 'Date' column
            list_months = []
            for i in range(df.shape[0]):
                list_months.append(df['Date'][i].month)

            # insert the values in a new column in df, called 'Month Value'
            df['Month Value'] = list_months

            # create a new feature called 'Day of the Week'
            df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())


            # drop the 'Date' column from df
            df = df.drop(['Date'], axis = 1)

            # re-order the columns in df
            column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                                'Transportation Expense', 'Distance to Work', 'Age',
                                'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                                'Pet', 'Absenteeism Time in Hours']
            df = df[column_names_upd]


            # map 'Education' variables; the result is a dummy
            df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

            # replace the NaN values
            df = df.fillna(value=0)

            # drop the variables we decide we don't need
            df = df.drop(['Absenteeism Time in Hours','Day of the Week','Daily Work Load Average','Distance to Work'],axis=1)

            #included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()

            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)

        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):
                pred = self.reg.predict_proba(self.data)[:,1] #taking only scaled data
                return pred

        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs

        # predict the outputs and the probabilities and
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data

