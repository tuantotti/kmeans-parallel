# K-means parallel
A parallel implementation of the unsupervised clustering algorithm K-means with **OpenMPI**. The parallelization leverages a message-passing protocol that allows communication among nodes (MPI).


## Requirements
The project requires:
- OpenMPI


## Compile and run
To compile it you need to run the following command: 
~~~~
mpic++ -o main main.cpp Node.cpp
~~~~
~~~~
mpirun -np (number of processors) ./main
~~~~
If you have configured a cluster of machines in order to run it among several nodes, such as master node, client1, and client2, the master node must run:
~~~~
mpirun -np (no. processors) -host master,client1,client2 ./main
~~~~
Or you can directly call a specific hostname configuration file:
~~~~
mpirun --hostfile /etc/hosts --oversubscribe -np (no. processors) main
~~~~

## Workflow
### Initial Configuration
Suppose we have P processors, where the master node is defined by the Rank 0, and a dataset with N points represented in the space with M-dimensional vectors. Then the master node: 
1. loads the dataset and scatters it among nodes, assigning to each of them N/P points. R Remaining points are assigned to the first R nodes. 
2. reads initial configuration parameters: no. clusters, no. dimensions, max iterations.
3. chooses K points as initial centroids and broadcast them to the other nodes.

### K-means 'loop'
4. In each node the following steps are executed: 
	- In each processor for each data point find membership using distance. (Euclidean, Cosine Similarity)
	- Recalculate local means for each cluster in each processor (recalculate centroids).
	- Globally broadcast all local means for each processor to find the global mean.

<p align="center">
<img width="80%" src="https://github.com/tuantotti/kmeans-parallel/blob/main/img/MPI_Allreduce.png"/>
</p>

5. Once we get the local summations, with the MPI_Allreduce operation, we can store the sum of the local summations (global sum) in each node. In the same way, we can obtain the global number of points in each cluster and store that value in each node. So to recalculate a centroid, we can simply divide the global sum of that cluster over the number of points belonging to it. Compute new centroids
6. Go to point 4 and repeat until termination

## Distance Metrics
Two distance metrics are implemented: 
1. Euclidean Distance
2. Cosine Similarity

## Termination
K-means converge when no more point changes its membership status. Since our dataset is distributed among nodes, we cannot know directly if no more changes occur. 
To deal with this, we have defined a flag in each node that is set when no more changes in the local dataset happen. Those flags are collected by nodes to check if there are changes or not. Furthermore, in order to avoid unnecessary long computation, we have limited the number of iterations that can be executed. 


