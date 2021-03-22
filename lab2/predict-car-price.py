import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
        	self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def prepare_X(df):
        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        df_num = df[base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

    def linear_regression(X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]


    def validate(self):
        np.random.seed(2)
        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)
        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]
        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()
        y_train = np.log1p(df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        y_test = np.log1p(df_test.msrp.values)

        X_train = CarPrice.prepare_X(df_train)
        w_0, w = CarPrice.linear_regression(X_train, y_train)
        pred_train = w_0 + X_train.dot(w)
        X_val = CarPrice.prepare_X(df_val)
        pred_val = w_0 + X_val.dot(w)
        X_test = CarPrice.prepare_X(df_test)
        pred_test = w_0 + X_test.dot(w)
        #Printing car details and predicted prices from test set
        df_test['msrp_pred']=np.expm1(pred_test)
        dfn=df_test[['engine_cylinders', 'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity', 'msrp', 'msrp_pred']].head(5)
        print(dfn.to_markdown(tablefmt="grid"))
	

if __name__ == "__main__":
    carprice = CarPrice()
    carprice.trim()
    carprice.validate()