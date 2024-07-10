from river import anomaly
from river import compose
from river import datasets
from river import metrics
from river import preprocessing
from river import feature_extraction as fx
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import numpy as np


class AnomalyDetectionoSVM:

    def __init__(self, dataset, nu_value, umbral_score):
        self.nu_value = nu_value
        self.model =anomaly.OneClassSVM(nu=self.nu_value)
        self.dataset = dataset
        self.umbral_score = umbral_score


    def train(self):
        # Divide dataset into train and test
        # Obtain 500 samples for training that contains benign samples remove from the dataset
        # Select first 500 rows where 'Label' is 0
        dataset_train = self.dataset.loc[self.dataset['Label'] == 0].iloc[:500]

        # Drop these rows from the original dataset
        dataset_test = self.dataset.drop(dataset_train.index)
        # Drop benign samples from the test dataset
        dataset_test = dataset_test.drop(dataset_test.loc[dataset_test['Label'] == 0].iloc[:4494].index)

        # Shuffle dataset_test
        dataset_test = dataset_test.sample(frac=1).reset_index(drop=True)

        # Separate the label
        dataset_train_no_labels = dataset_train.drop(columns=['Label'])
        dataset_test_no_labels = dataset_test.drop(columns=['Label'])

        # Traning phase

        # Create a list of scores
        for i, row in dataset_train_no_labels.iterrows():
            self.model.learn_one(row.to_dict())
            
        print("Traning phase completed")

        # Testing phase

        # Create a list of scores
        scores = []

        fp = 0
        fn = 0
        tp = 0
        tn = 0
        

        for idx in dataset_test_no_labels.index:
            row = dataset_test_no_labels.loc[idx]
            score = self.model.score_one(row.to_dict())

            if score >self.umbral_score:
                self.model.learn_one(row.to_dict())  # Aprende solo de datos benignos
            label = dataset_test.loc[idx, "Label"]
            # print("Score: " + str(score) + " Label: " + str(label))
            if score < self.umbral_score and label == 1:
                tp += 1
            elif score > self.umbral_score and label == 0:
                tn += 1
            elif score < self.umbral_score and label == 0:
                fp += 1
            elif score > self.umbral_score and label == 1:
                fn += 1
                
            scores.append(score)

            # if (tp + fp) != 0 and (tp + tn) / (tp + tn + fp + fn) > 0.65:
            #     self.__create_plot(dataset_test, scores)

        return tp, tn, fp, fn


    def __create_plot(self, dataset_test, scores):
        # Convert dataframe to numoy array
        dataset_test = dataset_test.to_numpy()

        colors = []

        for score in scores:
            if score < self.umbral_score:
                colors.append('red')
            else:
                colors.append('blue')


        # Print in 3d

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dataset_test[:, 0], dataset_test[:, 1], dataset_test[:, 2], color=colors, s=3.0)


        # Cambia el punto de vista
        ax.view_init(elev=70, azim=-30)

        plt.title('oSVM (nu = ' + str(self.nu_value) + ', umbral = ' + str(self.umbral_score) + ')')    
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

        # Save in a file
        plt.savefig('oSVM/oSVM_' + str(self.nu_value)+ "_"+str(self.umbral_score) + '.png')



