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
    tittle = "Time with communication: K = " + K + ", N = " + N
    plt.figure(figsize=(16, 9))
    plt.plot(numProcesses, exeTimeWithCom)
    plt.scatter(numProcesses, exeTimeWithCom, label='Fix n')
    plt.title(tittle)
    plt.xlabel('N_CPU')
    plt.ylabel('Total Time With Tcom (s)')
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    # Show the plot
    plt.savefig('img/execution-time-with-communication.png')
    plt.show()

def display_graph_without_tcom():
    df_additonal = pd.read_csv('results/execution-time.csv', nrows=6)
    df_additonal.drop_duplicates(inplace=True)
    df = pd.read_csv('results/execution-time.csv', skiprows=6)

    numProcesses = df['Number of Processes']
    exeTimeWithoutCom = df['Total Time without Communication']

    # Create a line plot
    K = df_additonal.loc[0, 'Number of Clusters']
    N = df_additonal.loc[0, 'Total Points']
    tittle = "Time without communication: K = " + K + ", N = " + N
    plt.figure(figsize=(16, 9))
    plt.plot(numProcesses, exeTimeWithoutCom)
    plt.scatter(numProcesses, exeTimeWithoutCom, label='Fix n')
    plt.title(tittle)
    plt.xlabel('N_CPU')
    plt.ylabel('Total Time Without Tcom (s)')
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    # Show the plot
    plt.savefig('img/execution-time-without-communication.png')
    plt.show()

def display_graph_fix_cpu_with_tcom():
    df_additonal = pd.read_csv('results/test.csv', nrows=2)
    df_additonal.drop_duplicates(inplace=True)
    df = pd.read_csv('results/test.csv', skiprows=2)

    numPoints = df['Number of Points']
    exeTimeWithCom = df['Total Time with Communication']

    # Create a line plot
    K = df_additonal.loc[0, 'Number of Clusters']
    N_CPU = df_additonal.loc[0, 'Number of Processes']
    tittle = "Time with communication: K = " + K + ", N_CPU = " + N_CPU
    plt.figure(figsize=(16, 9))
    plt.plot(numPoints, exeTimeWithCom)
    plt.scatter(numPoints, exeTimeWithCom, label='Fix N_CPU')
    plt.title(tittle)
    plt.xlabel('Number of Points')
    plt.ylabel('Total Time With Tcom (s)')
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    # Show the plot
    plt.savefig('img/fix-CPU-communication.png')
    plt.show()

def display_graph_fix_cpu_without_tcom():
    df_additonal = pd.read_csv('results/test.csv', nrows=2)
    df_additonal.drop_duplicates(inplace=True)
    df = pd.read_csv('results/test.csv', skiprows=2)

    numPoints = df['Number of Points']
    exeTimeWithCom = df['Total Time without Communication']

    # Create a line plot
    K = df_additonal.loc[0, 'Number of Clusters']
    N_CPU = df_additonal.loc[0, 'Number of Processes']
    tittle = "Time without communication: K = " + K + ", N_CPU = " + N_CPU
    plt.figure(figsize=(16, 9))
    plt.plot(numPoints, exeTimeWithCom)
    plt.scatter(numPoints, exeTimeWithCom, label='Fix N_CPU')
    plt.title(tittle)
    plt.xlabel('Number of Points')
    plt.ylabel('Total Time With Tcom (s)')
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    # Show the plot
    plt.savefig('img/fix-CPU-without-communication.png')
    plt.show()

if __name__ == '__main__':
    # display_graph_with_tcom()
    # display_graph_without_tcom()
    display_graph_fix_cpu_with_tcom()
    display_graph_fix_cpu_without_tcom()