import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def display_graph_with_tcom():
    df_additonal = pd.read_csv('results/execution-time.csv', nrows=6)
    df_additonal.drop_duplicates(inplace=True)
    df = pd.read_csv('results/execution-time.csv', skiprows=6)

    numProcesses = df['Number of Processes']
    exeTimeWithCom = df['Total Time with Communication']

    # Create a line plot
    K = df_additonal.loc[0, 'Number of Clusters']
    N = df_additonal.loc[0, 'Total Points']
    tittle = "K = " + K + ", N = " + N
    plt.figure(figsize=(16, 9))
    plt.plot(numProcesses, exeTimeWithCom)
    plt.title(tittle)
    plt.xlabel('N_CPU')
    plt.ylabel('Total Time With Tcom (s)')
    # Show the plot
    plt.savefig('img/execution-time-with-communication.png')
    plt.show()

def display_graph_without_tcom():
    df_additonal = pd.read_csv('results/execution-time.csv', nrows=6)
    df_additonal.drop_duplicates(inplace=True)
    df = pd.read_csv('results/execution-time.csv', skiprows=6)

    numProcesses = df['Number of Processes']
    exeTimeWithCom = df['Total Time without Communication']

    # Create a line plot
    K = df_additonal.loc[0, 'Number of Clusters']
    N = df_additonal.loc[0, 'Total Points']
    tittle = "K = " + K + ", N = " + N
    plt.figure(figsize=(16, 9))
    plt.plot(numProcesses, exeTimeWithCom)
    plt.title(tittle)
    plt.xlabel('N_CPU')
    plt.ylabel('Total Time Without Tcom (s)')
    # Show the plot
    plt.savefig('img/execution-time-without-communication.png')
    plt.show()

if __name__ == '__main__':
    display_graph_with_tcom()
    display_graph_without_tcom()