import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
    res = pd.read_csv('results/point-saved.csv')

    res['Cluster_id'].value_counts().plot.bar(x='Cluster Id', y='Number of Points', rot=0)
    plt.show()
    print(res['Cluster_id'].value_counts())

    cluster_df = res.groupby('Cluster_id')['Point_id'].apply(list)

    for index, cluster in enumerate(cluster_df):

        print('----- CLUSTER {} -----'.format(index))
        print(len(cluster))
