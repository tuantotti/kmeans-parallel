import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

np.random.seed(11)
means = [[2, 2], [10, 3], [3, 10]]
cov = [[1, 0], [0, 1]]
N = 60000
K = 3

def create_dataset(K, N, means, cov, is_save=False):
    # create distribution around 3 points in means
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)

    X = np.concatenate((X0, X1, X2), axis = 0)
    print(X.shape)
    df = pd.DataFrame(X)

    if is_save:
        with open('data/random-dataset-2-180000.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([2, 3, 1000])
            csvwriter.writerows(df.values.tolist())


    print(df.head())
    def kmeans_display(X):
        plt.plot(X[:, 0], X[:, 1], 'o', markersize = 4)
        plt.axis('equal')
        plt.plot()
        plt.show()

    kmeans_display(X)

create_dataset(K=K, N=N, means=means, cov=cov, is_save=True)