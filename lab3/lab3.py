import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

        
class DiabetesClassifier:

    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        # print(self.pima.head())
        self.X_test = None
        self.y_test = None

    def define_features(self, feature_cols):
        """
        Defines the features to use from the supplied data set.
        :param feature_cols: the columns to use for constructing the feature matrix
        :return: the feature matrix X and the label vector y
        """
        self.X = self.pima[feature_cols]
        self.y = self.pima.label

    def train(self, split_size, shuff):
        """
        Trains the classification model. Applies a standard scaler to the feature matrix,
        splits the data, then trains the model using the training data.
        :param split_size: the test size to partition the data set to
        :param shuff: True=shuffle, False=don't shuffle when the data set is split
        :return: the trained logistic regression model
        """
        # scale the features before training the model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # partition the data set into training and test sets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X_scaled, self.y, test_size=split_size, shuffle=shuff, random_state=0)

        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg

    def predict(self, model):
        """
        Generates predictions using the test data set generated during training.
        :return: the predictions vector, the accuracy of the predictions, and the confusion matrix
        """
        y_pred_class = model.predict(self.X_test)
        accuracy = self.calculate_accuracy(y_pred_class)
        conf_matrix = self.confusion_matrix(y_pred_class)
        return y_pred_class, accuracy, conf_matrix

    def calculate_accuracy(self, result):
        """
        Calculates the accuracy of the predictions as compared to the test label data.
        :param result: the predictions vector that will be evaluated
        :return: the accuracy score
        """
        return metrics.accuracy_score(self.y_test, result)

    def examine(self):
        """
        Outputs some distribution info about the test data set.
        :return: the test data mean
        """
        dist = self.y_test.value_counts()
        print('Distribution:')
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        print(f"% of ones: {percent_of_ones}")
        print(f"% of zeros: {percent_of_zeros}")
        return self.y_test.mean()

    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)

def extract_conf_matrix_elements(conf_matrix):
    """
    Extracts the components of the supplied confusion matrix.
    :return: The true-positive(tp), false-positive(fp), false-negative(fn), and true-negative(tn) values
    """
    tp = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tn = conf_matrix[1][1]
    return tp,fp,fn,tn

def display_header():
    """
    Displays the header for the results to display.
    """
    print('| %-30s | %-30s | %-30s | %-30s |' % ('Experiment', 'Accuracy', 'Confusion Matrix', 'Comment'))
    print('|', end='')
    for i in range(131):
        print('-', end='')
    print('|\n', end='')

def display_results(results):
    """
    Displays the results from a test run, which specifies the Experiment, Accuracy, Confusion Matrix, and Comment.
    """
    print('| ', end='')
    for x in results:
        print('%-30s' % x, end=' | ')
    print('\n', end='')

if __name__ == "__main__":

    classifier = DiabetesClassifier()
    display_header()

    # Baseline
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
    classifier.define_features(feature_cols)
    logreg = classifier.train(split_size=0.33, shuff=True)
    preds, accuracy, conf_matrix = classifier.predict(logreg)
    tp, fp, fn, tn = extract_conf_matrix_elements(conf_matrix)
    conf_matrix_str = f'[[{tp},{fp}], [{fn},{tn}]]'
    rez = (title, acc, conf, comments) = 'Baseline', accuracy, conf_matrix_str, 'baseline'
    display_results(rez)

    # Solution 1 : include glucose and pedigree as additional feature attributes
    feature_cols = ['pregnant', 'glucose', 'insulin', 'bmi', 'pedigree', 'age']
    classifier.define_features(feature_cols)    # set the features to use for training the model
    # train the model
    logreg = classifier.train(split_size=0.33, shuff=True)
    # generate predictions against the test data
    preds, accuracy, conf_matrix = classifier.predict(logreg)
    # output the results
    tp, fp, fn, tn = extract_conf_matrix_elements(conf_matrix)
    conf_matrix_str = f'[[{tp},{fp}], [{fn},{tn}]]'
    rez = (title, acc, conf, comments) = 'Solution 1', accuracy, conf_matrix_str, 'Include glucose and pedigree'
    display_results(rez)

    # Solution 2: include all columns as feature attributes
    feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    classifier.define_features(feature_cols)    # set the features to use for training the model
    # train the model
    logreg = classifier.train(split_size=0.33, shuff=True)
    # generate predictions against the test data
    preds, accuracy, conf_matrix = classifier.predict(logreg)
    # output the results
    tp, fp, fn, tn = extract_conf_matrix_elements(conf_matrix)
    conf_matrix_str = f'[[{tp},{fp}], [{fn},{tn}]]'
    rez = (title, acc, conf, comments) = 'Solution 2', accuracy, conf_matrix_str, 'Include all columns'
    display_results(rez)

    # Solution 3: include all columns as feature attributes
    feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    classifier.define_features(feature_cols)    # set the features to use for training the model
    # train the model
    logreg = classifier.train(split_size=0.33, shuff=False)
    # generate predictions against the test data
    preds, accuracy, conf_matrix = classifier.predict(logreg)
    # output the results
    tp, fp, fn, tn = extract_conf_matrix_elements(conf_matrix)
    conf_matrix_str = f'[[{tp},{fp}], [{fn},{tn}]]'
    rez = (title, acc, conf, comments) = 'Solution 3', accuracy, conf_matrix_str, 'Disable shuffle during split'
    display_results(rez)