from lionets.LioNexplainers import LioNexplainer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns

class LioNet:
    def __init__(self, model=None, autoencoder=None, decoder=None, encoder=None, feature_names=None):
        self.model = model
        self.autoencoder = autoencoder
        self.decoder = decoder
        self.encoder = encoder
        self.feature_names = feature_names
        self.generated_neighbourhood = []
        self.accuracy = 0

    def explain_instance(self, new_instance):
        self.instance = new_instance
        encoded_instance = self.encoder.predict(list(self.instance))
        neighbourhood = self.neighbourhood_generation(encoded_instance)
        self.final_neighbourhood = self.decoder.predict([neighbourhood])
        print(self.model.predict(encoded_instance))
        print("The predictor classified:",self.model.predict(encoded_instance)[0])
        self.neighbourhood_targets = self.model.predict(self.encoder.predict(self.final_neighbourhood))
        explainer = LioNexplainer(Ridge(), self.instance, self.final_neighbourhood, self.neighbourhood_targets, self.feature_names)
        explainer.fit_explanator()
        explainer.print_fidelity()
        explainer.show_explanation()
        return True

    def neighbourhood_generation(self, encoded_instance):
        instance = encoded_instance[0]
        instance_length = len(instance)
        local_neighbourhood = []
        for i in range(0, instance_length):
            gen1 = [0] * instance_length
            gen1[i] = instance[i]  # Only one feature
            gen2 = instance.copy() # Removing low valued features and increasing high value
            if gen2[i] < 0.2:
                gen2[i] = 0
            elif gen2[i] > 0.2:
                gen2[i] = gen2[i] * 5
            gen3 = instance.copy()  # Enhancing one feature
            gen3[i] = gen3[i] * 10
            gen4 = instance.copy()  # Removing one feature
            gen4[i] = 0
            gen5 = instance.copy()  # Enhancing low valued features a bit
            if gen5[i] < 0.02:
                gen5[i] = gen5[i] * 2
            else:
                gen5[i] = gen5[i]
            #local_neighbourhood.append(list(gen1))
            #local_neighbourhood.append(list(gen2))
            local_neighbourhood.append(list(gen3))
            local_neighbourhood.append(list(gen4))
            #local_neighbourhood.append(list(gen5))
        local_neighbourhood.append(instance)
        return local_neighbourhood

    def print_neighbourhood_labels_distribution(self):
        # matplotlib histogram
        plt.hist(self.neighbourhood_targets, color='blue', edgecolor='black',
                 bins=int(180 / 5))

        # seaborn histogram
        sns.distplot(self.neighbourhood_targets, hist=True, kde=False,
                     bins=int(180 / 5), color='blue',
                     hist_kws={'edgecolor': 'black'})
        # Add labels
        plt.title('Histogram of neighbourhood probabilities')
        plt.ylabel('Neighbours')
        plt.xlabel('Prediction Probabilities')
        plt.show()
