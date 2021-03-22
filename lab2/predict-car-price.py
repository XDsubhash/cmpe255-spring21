import numpy as np
import pandas as pd


def revert_log_scaling(df):
    return df.head().applymap(np.expm1)


# noinspection PyPep8Naming
def prepare_X(df):
    df = df.copy()
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)',
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)

    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)

    for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheens_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)

    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)

    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)

    for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)

    df_num = df[features]

    # df_num.fillna(0)
    # X = df_num.values
    # return X
    return df_num.fillna(0)


def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'{len(self.df)} lines loaded')

        self.trim()

        np.random.seed(2)

        n = len(self.df)

        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        self.df_train = df_shuffled.iloc[:n_train].copy()
        self.df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        self.df_test = df_shuffled.iloc[n_train+n_val:].copy()

        # self.y_train = np.log1p(self.df_train.msrp.values)
        # self.y_val = np.log1p(self.df_val.msrp.values)
        # self.y_test = np.log1p(self.df_test.msrp.values)

        self.y_train = self.df_train[["msrp"]].applymap(np.log1p)
        self.y_val = self.df_val[["msrp"]].applymap(np.log1p)
        self.y_test = self.df_test[["msrp"]].applymap(np.log1p)

        del self.df_train['msrp']
        del self.df_val['msrp']
        del self.df_test['msrp']

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    @staticmethod
    def linear_regression_reg(X, y, r=0.0):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        reg = r * np.eye(XTX.shape[0])
        XTX = XTX + reg

        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]

    def train_reg(self, r):
        X_train = prepare_X(self.df_train)
        return self.linear_regression_reg(X_train, self.y_train, r)

    def validate_reg(self, w_0, w):
        X_val = prepare_X(self.df_val)
        return w_0 + X_val.dot(w)

    def test_reg(self, w_0, w):
        X_test = prepare_X(self.df_test)
        return w_0 + X_test.dot(w)


def get_rmse_after_training_and_validating(r):
    w_0, w = car_price.train_reg(r)
    rmse_value = rmse(car_price.y_val.to_numpy(), car_price.validate_reg(w_0, w).to_numpy())
    print(f"r={r} produces rmse={rmse_value}")
    return rmse_value


if __name__ == '__main__':
    car_price = CarPrice()

    # Train & validate different r values to determine the one which produces the min rmse on the validation data set.
    r_which_produces_min_rmse_on_validation_data = min([0, 0.001, 0.01, 0.1, 1, 10], key=get_rmse_after_training_and_validating)
    print(f"r which produces min rmse on validation data: {r_which_produces_min_rmse_on_validation_data}")

    # Train using the r value which produces the min rmse on the validation data set.
    w_0, w = car_price.train_reg(r_which_produces_min_rmse_on_validation_data)

    # Test using the trained values of w_0 & w.
    y_test = car_price.y_test.head()
    y_pred = car_price.test_reg(w_0, w).head()

    # Format the output as directed.
    result = car_price.df_test[["engine_cylinders", "transmission_type", "driven_wheels", "number_of_doors", "market_category", "vehicle_size", "vehicle_style", "highway_mpg", "city_mpg", "popularity"]].head()
    result["msrp"] = revert_log_scaling(y_test)
    result["msrp_pred"] = revert_log_scaling(y_pred)
    pd.options.display.max_columns = None
    pd.options.display.width = 0
    print(result)