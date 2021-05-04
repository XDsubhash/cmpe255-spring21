import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from time import time
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Loading the dataset
def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    # introspect the images arrays to find the shapes (for plotting)
    n_samples= faces.images.shape 
    print(n_samples)
    return faces


faces = load_data()
faces


X = faces.data
n_features = X.shape[1]
n_samples,h,w= faces.images.shape

# the label to predict is the id of the person
y = faces.target
target_names = faces.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Crearting a pipeline for SVC using RandomizedPCA

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid= {'svc__C': [1,5,10,15,50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
clf =  GridSearchCV(model,param_grid)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
model = clf.best_estimator_
print("Best parameters found by grid search:")
print(clf.best_params_)

# Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set")
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=faces.target_names))

# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images,titles_pred,labels_actual, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        ax = plt.subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        fc = 'black'
        if titles_pred[i]!=labels_actual[i]:
          fc = 'red'
        title = "Predicted:" +titles_pred[i] + "\nActual:" + labels_actual[i] 
        plt.title(titles_pred[i], size=12,color=fc)
        plt.xticks(())
        plt.yticks(())
    fig.suptitle("Correct Predictions in black || Incorrect in Red "+'\n', fontsize=20)
    plt.show()

sample = np.random.choice(X_test.shape[0], size=24, replace=False)
images = X_test[sample]
labels_actual = y_test[sample]
labels_pred_sample = y_pred[sample]
names_pred = target_names[labels_pred_sample]
names_actual = target_names[labels_actual]

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

plot_gallery(images, names_pred, names_actual, h, w, n_row=4, n_col=6)

#Plotting the Confusion Matrix and Creating a heatmap

def heatmap(cmap):
  sns.heatmap(cmap,annot=True,xticklabels=target_names,yticklabels=target_names)
  plt.xlabel("Actual Values")
  plt.ylabel("Predicted Values")
  plt.show(block = True)

cmap = confusion_matrix(y_test, y_pred, labels=range(n_classes))
heatmap(cmap)