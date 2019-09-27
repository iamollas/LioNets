from LioNexplainers import LioNexplainer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import hamming_loss, r2_score
import operator


class LionForests:
    """Class for interpreting an instance"""
    def __init__(self, model=None, utilizer=None, feature_names=None, class_names=None):
        """Init function
        Args:
            model: The trained predictor model
            trees:
            feature_names: The selected features. The above networks have been trained with these.
            fidelity:
            accuracy:
        """
        self.model = model
        self.utilizer = utilizer
        self.trees = None
        if model is not None:
            self.trees = model.estimators_
        self.feature_names = feature_names
        self.class_names = class_names
        self.fidelity = 0
        self.accuracy = 0

    def train_text(self, train_data, train_target, params=None):
        random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)
        vec = TfidfVectorizer(analyzer='word')
        pipe = Pipeline(steps=[('vec', vec), ('random_forest', random_forest)])
        parameters = params
        if params is None:
            parameters = [{
                'vec__ngram_range': [(1, 1), (1, 2)], #, (1, 5)
                'vec__max_features': [5000],#, 10000, 50000, 100000
                'vec__stop_words': ['english', None],
                'random_forest__max_depth': [1, 10],#, 50, 100, 200
                'random_forest__max_features': [10, 50, None], #'sqrt', 'log2'
                'random_forest__bootstrap': [True, False],
                'random_forest__n_estimators': [500]#10, 100, 500, 1000
            }]
        clf = GridSearchCV(estimator=pipe, param_grid=parameters, cv=10, n_jobs=18, verbose=0, scoring='accuracy')
        clf.fit(train_data, train_target)
        self.accuracy = clf.best_score_
        self.feature_names = clf.named_steps['vec'].get_feature_names()
        self.model = clf.best_estimator_
        self.utilizer = clf.named_steps['vec']
        self.trees = self.model.estimators_

    def train_tab(self, train_data, train_target, scaling_method=None, feature_names=None, params = None):
        #BASED ON SCALING METHOD SCALE
        random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)
        parameters = params
        if params is None:
            parameters = [{
                'max_depth': [1, 10, 50, 100, 200],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'n_estimators': [10, 100, 500, 1000]
            }]
        clf = GridSearchCV(estimator=random_forest, param_grid=parameters, cv=10, n_jobs=18, verbose=0, scoring='accuracy')
        clf.fit(train_data, train_target)
        self.accuracy = clf.best_score_
        if feature_names is not None:
            self.feature_names = feature_names
        self.model = clf.best_estimator_
        self.utilizer = scaling_method #@!!!!WARNING HERE
        self.trees = self.model.estimators_

    def fidelity_between_nn_rf(self, data, nn_target):
        rf_target = self.model.predict(data)
        self.fidelity = hamming_loss(nn_target, rf_target)

    def local_tree_instance(self, instance, target, neighbours, targets):
        #UTILIZER IF ANY! TRANSFORM
        index = []
        for i in range(len(self.trees)):
            print(self.trees[i].predict([instance]))
            if target == self.trees[i].predict([instance])[0]:
                index.append(i)
        ranking = {}
        for i in index:
            rf_targets = self.trees[i].predict_proba(neighbours)#DO CALIBRATION
            ranking[i] = r2_score(target,rf_targets)
        best_tree = max(ranking.iteritems(), key=operator.itemgetter(1))[0]
        in_target = self.trees[best_tree].predict([instance])[0]
        path = self.trees[best_tree].decision_path([instance])
        return [self.trees[best_tree].predict([instance])[0], self.path_transformer(in_target, instance, path, best_tree)]

    def local_predict_instance(self, instance, target, neighbours, targets):
        return self.local_tree_instance(instance, target, neighbours, targets)[0]

    def local_explain_instance(self, instance, target, neighbours, targets):
        return self.local_tree_instance(instance, target, neighbours, targets)[1]


    def local_explain_through_paths(self, instance, target, neighbours, targets):
        for i in self.trees:
            print()

    def path_transformer(self, target, instance, path, best_tree):
        rule = "if "
        clauses_n_values1 = {}
        clauses_n_values2 = {}
        for node in path.indices:
            feature_id = self.trees[best_tree].tree_.feature[node]
            feature = self.feature_names[feature_id]
            threshold = self.trees[best_tree].tree_.threshold[node]
            if threshold!=-2.0:
                if(instance[feature_id] <= threshold):
                    clauses_n_values1.setdefault(feature, []).append(round(threshold,4))
                else:
                    clauses_n_values2.setdefault(feature, []).append(round(threshold,4))
        for k in clauses_n_values1:
            rule = rule + k + "<=" + str(min(clauses_n_values1[k])) + " and "
        for k in clauses_n_values2:
            rule = rule + k + ">" + str(max(clauses_n_values2[k])) + " and "
        if self.class_names is not None:
            rule = rule[:-4] + "then " + str(self.class_names[int(target)])
        else:
            rule = rule[:-4] + "then " + str(target)
        return rule



    def explain_instance(self, new_instance, normal_distribution=True):
        """Generates the explanation for an instance
        Args:
            new_instance: The instance to explain
            normal_distribution: Sets the distribution of the neighbourhood to be normal (In progress!)
        """
        self.instance = new_instance
        encoded_instance = self.encoder.predict(list(self.instance))
        neighbourhood = self.neighbourhood_generation(encoded_instance)
        import numpy as np
        self.final_neighbourhood = self.decoder.predict(np.array(neighbourhood))
        print("The predictor classified:",self.model.predict(self.instance)[0])
        self.neighbourhood_targets = self.model.predict(self.final_neighbourhood)
        if normal_distribution:
            self.neighbourhood_to_normal_distribution()
        explainer = LioNexplainer(Ridge(), self.instance, self.final_neighbourhood, self.neighbourhood_targets, self.feature_names)
        explainer.fit_explanator()
        explainer.print_fidelity()
        explainer.show_explanation()
        return True

    def neighbourhood_generation(self, encoded_instance):
        """Generates the neighbourhood of an instance
        Args:
            encoded_instance: The instance to generate neighbours
        Return:
            local_neighbourhood: The generated neighbours
        """
        instance = []
        for i in range(0, len(encoded_instance[0])):
            instance.append(encoded_instance[0][i])
        instance_length = len(instance)
        local_neighbourhood = []
        for i in range(0, instance_length): #Multiplying one feature value at a time with
            for m in [0.25,0.5,0,1,2]: # 1/4, 1/2, 0, 1, 2
                gen = instance.copy()
                gen[i] = gen[i] * m
                local_neighbourhood.append(list(gen))
                del gen
        for i in range(0,5):
            local_neighbourhood.append(instance)
        return local_neighbourhood + local_neighbourhood #We do this in order to have a bigger dataset. But there is no difference after all.

    #In Progress
    def neighbourhood_to_normal_distribution(self):
        """Transforms the distribution of the neighbourhood to normal
        """
        old_neighbourhood = self.final_neighbourhood
        old_targets = self.neighbourhood_targets.copy()
        #...

    def print_neighbourhood_labels_distribution(self):
        """Presenting in a plot the distribution of the neighbourhood data
        """
        plt.hist(self.neighbourhood_targets, color='blue', edgecolor='black', bins=int(180 / 5))
        sns.distplot(self.neighbourhood_targets, hist=True, kde=False, bins=int(180 / 5), color='blue', hist_kws={'edgecolor': 'black'})

        plt.title('Histogram of neighbourhood probabilities')
        plt.ylabel('Neighbours')
        plt.xlabel('Prediction Probabilities')
        plt.show()