#%%
from joblib import dump
import json
import os

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

data = "data"
model = "model_sklearn"

train_json = json.load(open(os.path.join(data, "train.json")))
test_json = json.load(open(os.path.join(data, "test.json")))

X = [" ".join(item["ingredients"]) for item in train_json]
Y = [item["cuisine"] for item in train_json]


pipeline = Pipeline(
    [("tfidf", TfidfVectorizer()), ("clf", TransformedTargetRegressor())]
)

parameters = [
    {
        "clf": (
            BernoulliNB(),
            DecisionTreeClassifier(),
            DummyClassifier(),
            ExtraTreeClassifier(),
            KNeighborsClassifier(),
            LinearSVC(),
            MultinomialNB(),
            PassiveAggressiveClassifier(),
            Perceptron(),
            RidgeClassifier(),
            SGDClassifier(),
            LogisticRegression(),
        )
    },
    {"clf": (SVC(),), "clf__C": (1, 10, 100, 1000)},
]


grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

_ = grid_search.fit(X, Y)


#%%
dev_means = grid_search.cv_results_["mean_test_score"]
dev_stds = grid_search.cv_results_["std_test_score"]
dev_params = grid_search.cv_results_["params"]

print("Development Set Results")
for mean, std, params in zip(dev_means, dev_stds, dev_params):
    print(f"  {mean:.3f} +/- {std*2:.3f} for {params['clf'].__class__.__name__}")

print(f"Best Score")
print(f"  {grid_search.best_score_:.3f}")

best_clf = grid_search.best_estimator_.named_steps["clf"]
print(f"Best Classifier")
print(f"  {best_clf.__class__.__name__}")

best_parameter_set = best_clf.get_params()
print("Best Parameter Set")
for param_name in best_parameter_set:
    print(f"  {param_name}: {best_parameter_set[param_name]}")

dump(grid_search.best_estimator_, model)
