import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def compare_to_sklearn(is_save_fig=False):
    df = pd.read_csv('data/random-dataset-2.csv', skiprows=1, header=None)
    point_df = pd.read_csv('results/point-saved-2.csv')

    kmeans = KMeans(n_clusters=3, random_state=42, init='random')
    kmeans.fit(df)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121)

    u_labels = np.unique(labels)
    for i in u_labels:
        ax.scatter(df.iloc[labels == i , 0] , df.iloc[labels == i , 1] , label = i)

    # Mark cluster centers
    # ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200)
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    ax.set_title('Kmeans sklearn')
    print(cluster_centers)


    # fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(122)

    print(df.shape)
    print(point_df.shape)
    print(point_df['Cluster_id'].value_counts())
    print(df.iloc[91, :])

    name = 0
    for grp_name, grp_idx in point_df.groupby('Cluster_id').groups.items():
        # print(grp_idx)
        x = df.iloc[grp_idx,0]
        y = df.iloc[grp_idx,1]
        ax.scatter(x, y, label=name)  # this way you can control color/marker/size of each group freely
        # ax.scatter(df.iloc[grp_name, 0], df.iloc[grp_name, 1], c='red', marker='X', s=200)
        name+=1

    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    ax.set_title('Kmeans open mpi')

    if is_save_fig:
        plt.savefig('img/compare-to-sklearn.png')
    plt.show()

if __name__ == '__main__':
    compare_to_sklearn(is_save_fig=True)
