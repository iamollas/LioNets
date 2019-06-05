import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge


class LioNexplainer:
    def __init__(self, explanator=Ridge(), instance=None, train_data=None, target_data=None, feature_names=None):
        self.explanator = explanator
        self.instance = instance
        self.train_data = train_data
        self.target_data = target_data
        self.feature_names = feature_names
        self.accuracy_r2 = 0
        self.accuracy_r2 = 0

    def fit_explanator(self):
        self.explanator.fit(self.train_data, self.target_data)
        self.accuracy_r2 = r2_score(self.target_data, self.explanator.predict(self.train_data))

    def print_fidelity(self):
        print("The fidelity of the LioNet in terms of R^2 Score is:", self.accuracy_r2)
        print("The fidelity of the LioNet in terms of R^2 Score is:", self.accuracy_r2)

    def show_explanation(self):
        #if (str(type(self.explanator)) == "<class 'sklearn.linear_model.ridge.Ridge'>"):
        weights = self.explanator.coef_
        model_weights = pd.DataFrame({'features': list(self.feature_names), 'weights': list(weights[0] * self.instance.A[0])})
        model_weights = model_weights.sort_values(by='weights', ascending=False)
        model_weights = model_weights[(model_weights['weights'] != 0)]
        #model_weights = pd.concat([model_weights.head(5), model_weights.tail(5)])
        plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        sns.barplot(x="weights", y="features", data=model_weights)
        plt.xticks(rotation=90)
        plt.show()
        # Different Ifs for different types of explainers.
