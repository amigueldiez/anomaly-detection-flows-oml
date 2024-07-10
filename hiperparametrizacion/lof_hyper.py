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
import random
import time


class AnomalyDetectionLOF:
    
    def __init__(self, dataset, number_neighbors, q_value, protect_anomaly_detector):
        self.number_neighbors = number_neighbors
        self.q_value = q_value
        self.protect_anomaly_detector = protect_anomaly_detector

        self.model = anomaly.QuantileFilter(
            anomaly.LocalOutlierFactor(n_neighbors=self.number_neighbors),
            q=self.q_value,
            protect_anomaly_detector=self.protect_anomaly_detector
        )

        self.dataset = dataset
    

    def train(self):
        # Divide dataset into train and test
        # Obtain 10000 samples for training that contains benign samples remove from the dataset
        # Select first 10000 rows where 'Label' is 0
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
        anomalies = []

        accuracies = []

        fp = 0
        fn = 0
        tp = 0
        tn = 0

        print("Starting testing phase")
        for idx in dataset_test_no_labels.index:
            row = dataset_test_no_labels.loc[idx]
            if idx % 1000 == 0:
                print(self.q_value,"/",self.number_neighbors," Row: ", idx)

            score = self.model.score_one(row.to_dict())
            is_anomaly = self.model.classify(score)
            anomalies.append(is_anomaly)
            if not is_anomaly:
                self.model.learn_one(row.to_dict())  # Aprende solo de datos benignos si lo deseas
            label = dataset_test.loc[idx, "Label"]
            
            if is_anomaly and label == 1:
                tp += 1
            elif not is_anomaly and label == 0:
                tn += 1
            elif is_anomaly and label == 0:
                fp += 1
            elif not is_anomaly and label == 1:
                fn += 1

        print("Testing phase completed")

        # Create a plot
        self.__create_plot(dataset_test, anomalies)


        # Return the metrics
        return tp, tn, fp, fn


    def test(self):
        num = random.randint(0, 25)
        time.sleep(num)  # Sleep for a random time between 0 and 25 seconds
        return 2,3,4,5

    def __create_plot(self, dataset_test, anomalies):
        # Convert dataframe to numoy array
        dataset_test = dataset_test.to_numpy()

        colors = []

        for anomaly in anomalies:
            if anomaly:
                colors.append('red')
            else:
                colors.append('blue')

        # Print in 3D

        

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dataset_test[:, 0], dataset_test[:, 1], dataset_test[:, 2], color=colors, s=3.0)


        # Cambia el punto de vista
        ax.view_init(elev=70, azim=-30)



        plt.title('LOF (q=' + str(self.q_value) + ', n_neighbors=' + str(self.number_neighbors) + ', protect_anomaly_detector=' + str(self.protect_anomaly_detector) + ')')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

        # Save in a file
        plt.savefig('lof/lof_' + str(self.q_value) + '_' + str(self.number_neighbors) + '_' + str(self.protect_anomaly_detector) + '.png')
