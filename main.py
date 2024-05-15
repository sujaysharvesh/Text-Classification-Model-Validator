import pandas as pd
import numpy as np
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def generate_corpus(dataset):
    corpus = []
    for i in range(0, dataset.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        allwords = stopwords.words('english')
        allwords.remove('not')
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(allwords)]
        review = ' '.join(review)
        corpus.append(review)

    count_vect = CountVectorizer(max_features=None)
    X = count_vect.fit_transform(corpus).toarray()
    Y = dataset.iloc[:, -1].values

    return X, Y


def split_dataset(X, Y, random_state=0, test_size=0.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=random_state, test_size=test_size)

    return X_train, X_test, Y_train, Y_test


def classification_model(X_train, X_test, Y_train, Y_test, model, params, metrics: list, save=False) -> list:
    mod = eval(model)(**params)
    mod.fit(X_train, Y_train)

    Y_pred = mod.predict(X_test)
    perf = {i: eval(i)(Y_test, Y_pred) for i in metrics}
    score = pd.Series(list(perf.values()), index=list(perf.keys()), name=model)

    if save == True:
        joblib.dump(value=mod, filename='C:/Users/sujay/Downloads/ML/model.pkl')

    return pd.DataFrame(score).T


def hyperparameter_tuning(X_train, Y_train):
    classifiers = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'param_grid': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(),
            'param_grid': {'criterion': ['gini', 'entropy'],
                           'max_depth': [None, 10, 20],
                           'min_samples_split': [2, 5, 10],
                           'min_samples_leaf': [1, 2, 4],
                           'max_features': ['auto', 'sqrt', 'log2'],
                           'splitter': ['best', 'random']
                           }
        },

        'SVC': {
            'model': SVC(),
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(),
            'param_grid': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }
        },
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'param_grid': {}
        }
    }

    results = {}
    for name, config in classifiers.items():
        model = config['model']
        param_grid = config['param_grid']

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, Y_train)
        best_params_grid = grid_search.best_params_

        results[name] = {
            'best_params': best_params_grid
        }
    return results


if __name__ == '__main__':
    # Data prep
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3, encoding='unicode_escape')
    X, Y = generate_corpus(dataset=dataset)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    # Classification model execution

    result = hyperparameter_tuning(X_train, Y_train)
    data = []

    # Append model name and best parameters as dictionary
    for name, best_params in result.items():
        data.append({'model': name, 'params': best_params['best_params']})

    final_result = []

    # Iterate over the list of model names and parameters
    for item in data:
        model = item['model']
        params = item['params']

        # Call classification_model function with the model name and parameters
        finals = classification_model(X_train, X_test, Y_train, Y_test, model=model, params=params,
                                      metrics=['accuracy_score', 'precision_score', 'recall_score', 'f1_score'],
                                      save=True)
        final_result.append(finals)

    final_score = pd.concat(final_result)
    print(final_score)
