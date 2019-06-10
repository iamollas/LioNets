import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge


class LioNexplainer:
    def __init__(self, explanator=Ridge(), instance=None, train_data=None, target_data=None, feature_names=None):
        self.explanator = explanator
        self.instance = instance
        self.train_data = train_data
        self.target_data = target_data
        self.feature_names = feature_names
        self.accuracy_r2 = 0
        self.accuracy_mse = 0

    def fit_explanator(self):
        self.explanator.fit(self.train_data, self.target_data)
        self.accuracy_r2 = r2_score(self.target_data, self.explanator.predict(self.train_data))
        self.accuracy_mse = mean_squared_error(self.target_data, self.explanator.predict(self.train_data))
    def print_fidelity(self):
        print("The fidelity of the LioNet in terms of R^2 Score is:", self.accuracy_r2)
        print("The fidelity of the LioNet in terms of Mean Square Error is:", self.accuracy_mse)

    def show_explanation(self):
        #if (str(type(self.explanator)) == "<class 'sklearn.linear_model.ridge.Ridge'>"):
        weights = self.explanator.coef_
        model_weights = pd.DataFrame({"Instance's Features": list(self.feature_names), "Features' Contribution": list(weights[0] * self.instance.A[0])})
        model_weights = model_weights.sort_values(by="Features' Contribution", ascending=False)
        model_weights = model_weights[(model_weights["Features' Contribution"] != 0)]
        #model_weights = pd.concat([model_weights.head(5), model_weights.tail(5)])
        plt.figure(num=None, figsize=(6, 6), dpi=200, facecolor='w', edgecolor='k')
        sns.barplot(x="Features' Contribution", y="Instance's Features", data=model_weights)
        plt.xticks(rotation=90)
        plt.show()
        # Different Ifs for different types of explainers.
