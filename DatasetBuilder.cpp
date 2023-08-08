#include "DatasetBuilder.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <cstdlib>
#include <ctime>

DatasetBuilder::DatasetBuilder(int numPoints, int pointDimension, int numClusters, int maxIteration, string filename) {
    this->numPoints = numPoints;
    this->pointDimension = pointDimension;
    this->numClusters = numClusters;
    this->maxIteration = maxIteration;
    this->filename = filename;
}

DatasetBuilder::~DatasetBuilder() {}

void DatasetBuilder::createDataset(){
    srand(time(NULL));

    ofstream myfile;
    myfile.open("data/" + filename + ".csv");
    myfile << pointDimension << "," << numClusters << "," << maxIteration << "\n";
    for(int p = 0; p < numPoints; p++){
        for(int j = 0; j < pointDimension; j++){
            myfile << rand() % 300 + (-150) ;

            if(j < pointDimension-1){
                myfile << ",";
            }
        }
        myfile << "\n";
    }
    myfile.close();
}

