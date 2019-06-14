import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.linear_model import RidgeCV,PassiveAggressiveRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class LioNexplainer:
    """Class for interpreting an instance"""

    def __init__(self, explanator=RidgeCV(alphas=[0,1e-3, 1e-2, 1e-1, 1],fit_intercept=[True,False],cv=10), instance=None, train_data=None, target_data=None, feature_names=None):
        """Init function
        Args:
            explanator: The transparent model that is going to be used for the explanation. Default is Ridge Regression Algorithm
            instance: The instance to explain
            train_data: The neighbourhood of the above instance
            target_data: The predictions of the neural network for this neighbours
            feature_names: The selected features.
        """
        self.explanator = explanator
        self.instance = instance
        self.train_data = train_data
        self.target_data = target_data
        self.feature_names = feature_names
        self.accuracy_r2 = 0
        self.accuracy_mse = 0
        self.fidelity = 0

    def fit_explanator(self):
        """fit_explanator function trains the transparent regression model with the neighbourhood data
        """
        distances = []
        for i in self.train_data:
            distances.append(cosine_similarity([i],[self.instance.A[0]])[0][0]*100)

        self.explanator.fit(self.train_data, self.target_data)
        y_pred = self.explanator.predict(self.train_data)
        self.accuracy_r2 = r2_score(self.target_data, y_pred)
        self.accuracy_mse = mean_squared_error(self.target_data, y_pred)
        target_data_binary = []
        predicted_data_binary = []
        for i in y_pred:
            if i>0.5:
                predicted_data_binary.append(1)
            else:
                predicted_data_binary.append(0)
        for i in self.target_data:
            if i>0.5:
                target_data_binary.append(1)
            else:
                target_data_binary.append(0)
        self.fidelity = accuracy_score(target_data_binary,predicted_data_binary)

    #In progress!
    def print_fidelity(self):
        print("The fidelity of the LioNet in terms of Accuracy Score is:", self.fidelity)
        print("The fidelity of the LioNet in terms of R^2 Score is:", self.accuracy_r2)
        print("The fidelity of the LioNet in terms of Mean Square Error is:", self.accuracy_mse)

    def show_explanation(self):
        """show_explanation function extracts the weights for the features from the transparent trained model
        and it creates a plot explaining the weights for a specific instance.
        """
        #if (str(type(self.explanator)) == "<class 'sklearn.linear_model.ridge.Ridge'>"): #!In order to put more models except Ridge
        weights = self.explanator.coef_
        model_weights = pd.DataFrame({"Instance's Features": list(self.feature_names), "Features' Contribution": list(weights[0] * self.instance.A[0])})
        model_weights = model_weights.sort_values(by="Features' Contribution", ascending=False)
        model_weights = model_weights[(model_weights["Features' Contribution"] != 0)]
        plt.figure(num=None, figsize=(6, 6), dpi=200, facecolor='w', edgecolor='k')
        sns.barplot(x="Features' Contribution", y="Instance's Features", data=model_weights)
        plt.xticks(rotation=90)
        plt.show()