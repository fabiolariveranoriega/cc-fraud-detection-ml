from abc import ABC, abstractmethod

import pandas as pd
from sklearn import preprocessing


class GraphData(ABC):
    def __init__(self, path:str):
        super().__init__()
        self.data = pd.read_csv(path)


class CreditCardData(GraphData):
    def __init__(self, path):
        super().__init__(path)

    def preprocess(self):
        merchants_with_multiple_states = self.data.groupby('Merchant Name')['Merchant City'].nunique()
        merchants_with_multiple_states = merchants_with_multiple_states[merchants_with_multiple_states > 1]

        self.data['Merchant Name & City'] = self.data['Merchant Name'].astype(str)+ '_' + self.data['Merchant City']
        self.data['Merchant Name & City'].head(1)
        self.data['Amount'] = self.data['Amount'].replace('[\$,]', '', regex=True).astype(float)
        self.data[['hour', 'minute']] = self.data['Time'].str.split(':', expand=True).astype(int)
        self.data.drop(['Merchant State', 'Zip'], axis = 1, inplace = True)
        self.data['Errors?'].fillna(value = 'None', inplace = True)


        self.data.drop(['Time', 'Merchant Name', 'Merchant City'], axis = 1, inplace = True)


        # encoding 
        label_encoder = preprocessing.LabelEncoder()
        self.data['Use Chip'] = label_encoder.fit_transform(self.data['Use Chip'])


        label_encoder = preprocessing.LabelEncoder()
        self.data['Errors?'] = label_encoder.fit_transform(self.data['Errors?'])

        label_encoder = preprocessing.LabelEncoder()
        self.data['Is Fraud?'] = label_encoder.fit_transform(self.data['Is Fraud?'])

        user_ids = self.data['User'].unique()
        merchant_ids = self.data['Merchant Name & City'].unique()

        # Create dictionaries to map users and merchants to unique indices
        self.user_mapping = {user: idx for idx, user in enumerate(user_ids)}
        self.merchant_mapping = {merchant: idx + len(user_ids) for idx, merchant in enumerate(merchant_ids)}

    def get_mappings(self):
        return self.user_mapping, self.merchant_mapping



