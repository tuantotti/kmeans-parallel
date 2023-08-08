//
// Created by Claudio Russo Introito on 2019-04-28.
//

#ifndef KMEANSCLUSTERING_NODE_H
#define KMEANSCLUSTERING_NODE_H

#include <string>
#include <mpi.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include "Point.h"

using namespace std;


class Node{

private:
    int rank;
    int numProcesses;
    MPI_Comm comm;
    MPI_Datatype pointType;

    int total_values;
    int num_local_points;
    int K, max_iterations;
    int numPoints; // Total number of points in the whole dataset

    int notChanged; 
    int* memCounter; // Membership count
    int lastIteration;
    bool newDatasetCreated;
    string newDatasetFilename;
    int distance;   //Integer which refers to the number of the chosen distance by the user among: 1) Euclidean Distance  2) Cosine Similarity

    double* reduceArr;
    double* reduceResults;

    vector<Point> dataset; // The dataset at master node or process with rank id = 0
    vector<Point> localDataset; // the local dataset in each node
    vector<Point> clusters; // centroids
    vector<Point> localSum;
    int numPointsPerNode;
    vector<int> memberships; //This vector has same length as localDataset: for each point in localDataset is
                            // associated the id of nearest cluster in the corresponding position in membership
    int* globalMembership;
    double total_time; // total time with communication in each node
    double total_time_without_comm; // total time without communication in each node


    int getIdNearestCluster(Point p);
    void updateLocalSum();

public:
    Node(int rank, MPI_Comm comm = MPI_COMM_WORLD);
    ~Node();
    int getMaxIterations();
    void readDataset();
    void createDataset();
    void scatterDataset();
    void initCentroids();
    int run(int it);
    void computeGlobalMembership();
    int getNumPoints();
    int* getGlobalMemberships();
    void printClusters();
    void writeClusterMembership(string filename);

    void getStatistics();
    vector<double> SSW();   //Variance within cluster (https://math.stackexchange.com/questions/1009297/variances-for-k-means-clustering)
    double SSB();
    double squared_norm(Point p1, Point p2);
    double cosine_similarity(Point p1, Point p2);
    void setLastIteration(int lastIt);
    void setNumProcesses(int _numProcesses);
};


#endif
