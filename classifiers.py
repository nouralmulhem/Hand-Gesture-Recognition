from os import stat
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle

class Classifiers:

    @staticmethod
    def KNN_Classifier(features_arr, Y_train, features_test_arr, k = 11):
        clf = KNeighborsClassifier(n_neighbors=k)
        tree = clf.fit(features_arr, Y_train)
        results = tree.predict(features_test_arr)
        return results, tree   

    @staticmethod
    def SVM_Classifier(features_arr, Y_train, features_test_arr, gamma = 50, C=100):
        clf = svm.SVC( probability = True, gamma = gamma, C = C) 
        clf.fit(features_arr, Y_train)
        results_svm = clf.predict_proba(features_test_arr)
        final_res = np.argmax(results_svm, axis=1)+1
        filename = 'svm_model.sav'
        # pickle.dump(clf, open(filename, 'wb'))
        return results_svm, final_res, clf
       
    @staticmethod
    def Random_Forest_Classifier(features_arr, Y_train, features_test_arr, max_depth=20, random_state=0):
        clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
        clf.fit(features_arr, Y_train)
        results_rnd_forest = clf.predict_proba(features_test_arr)
        final_res = np.argmax(results_rnd_forest, axis=1)+1
        filename = 'rnd_forest_model.sav'
        # pickle.dump(clf, open(filename, 'wb'))
        return results_rnd_forest, final_res, clf

    @staticmethod
    def MajorityVote(features_arr, Y_train, features_test_arr):
        _, clf1 = Classifiers.KNN_Classifier(features_arr, Y_train, features_test_arr)
        # _, _, clf1 = Classifiers.MLPClassifier(features_arr, Y_train, features_test_arr)
        _, _, clf2 = Classifiers.SVM_Classifier(features_arr, Y_train, features_test_arr)
        _, _, clf3 = Classifiers.Random_Forest_Classifier(features_arr, Y_train, features_test_arr)

        eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
        eclf1 = eclf1.fit(features_arr, Y_train)
        filename = 'maj_vote_model.sav'
        pickle.dump(eclf1, open(filename, 'wb'))
        prediction = eclf1.predict(features_test_arr)
        return prediction

    @staticmethod
    def MLPClassifier(features_arr, Y_train, features_test_arr, hd = 50, hd2 = 50):
        clf = MLPClassifier(random_state=1, max_iter=2000, alpha=0.001, hidden_layer_sizes = (hd, hd2)).fit(features_arr, Y_train)
        results_mlp = clf.predict_proba(features_test_arr)
        final_res = np.argmax(results_mlp, axis=1)+1
        return results_mlp, final_res, clf



        # @staticmethod
        # def k_means_classifier(x_train, x_test, n_clusters=3, random_state=0):
        #     clf = KMeans(n_clusters=n_clusters, random_state=random_state)
        #     clf.fit(x_train)
        #     results = clf.predict(x_test)
        #     results += 1
        #     return results

        # @staticmethod
        # def adaboost_classifier(x_train, y_train, x_test, n_estimators=100, random_state=0):
        #     clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
        #     clf.fit(x_train, y_train)
        #     results = clf.predict_proba(x_test)
        #     return results

        # @staticmethod
        # def nn_classifier(x_train, y_train, x_test,  hidden_layer_sizes, max_iter=300, random_state=0):
        #     clf = MLPClassifier(random_state=random_state, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
        #     clf.fit(x_train, y_train)
        #     results = clf.predict_proba(x_test)
        #     return results

        # @staticmethod
        # def gradient_classifier(x_train, y_train, x_test, n_estimators=100, random_state=0, learning_rate=1.0, max_depth=8):
        #     clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        #                                      random_state=random_state)
        #     clf.fit(x_train, y_train)
        #     results = clf.predict_proba(x_test)
        #     return results
