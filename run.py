import os
import re
import subprocess
import pandas as pd
import csv

num_processes = [16, 24, 32, 40, 64, 128]
# num_processes = range(4, 32, 2)
# compile code 
os.system("mpic++ -fopenmp -o main main.cpp Node.cpp")

additional_df = pd.DataFrame(columns=['Number of Clusters', 'Total Points', '0'])
df = pd.DataFrame(columns=['Number of Processes', 'Total Time with Communication', 'Total Time without Communication'])

# run with multiple processes
for i in num_processes:
    print('Run with number of processes = ', i)
    # result = subprocess.run(['mpirun','-hostfile', '/etc/hosts', '--oversubscribe', '-np', str(i), 'main'], capture_output=True, text=True)
    result = subprocess.run(['mpirun','--oversubscribe', '-np', str(i), 'main'], capture_output=True, text=True)
    # Display the result from the command line
    # print(result.stdout)

    # search in the log when we run the algrithm
    number_of_clusters_match = re.search(r'Number of clusters K: ([0-9.]+)', result.stdout)
    total_points_match = re.search(r'Total points: ([0-9.]+)', result.stdout)
    total_time_match = re.search(r'Total Time with Communication: ([0-9.]+)', result.stdout)
    total_time_without_match = re.search(r'Total Time without Communication: ([0-9.]+)', result.stdout)

    if number_of_clusters_match and total_points_match:
        number_of_clusters = float(number_of_clusters_match.group(1))
        total_points = float(total_points_match.group(1))

        row = {
            'Number of Clusters': number_of_clusters,
            'Total Points': total_points,
            '0': 0
        }
        dfTemp = pd.DataFrame([row])
        additional_df = pd.concat([additional_df, dfTemp], ignore_index=True)
    else:
        print("Not Found")

    if total_time_match and total_time_without_match:
        total_time = float(total_time_match.group(1))
        total_time_without_com = float(total_time_without_match.group(1))
        # print(f"Total Time with Communication: {total_time}")
        row = {
            'Number of Processes': i,
            'Total Time with Communication': total_time,
            'Total Time without Communication': total_time_without_com
        }
        dfTemp = pd.DataFrame([row])
        df = pd.concat([df, dfTemp], ignore_index=True)
    else:
        print("Not Found")

# reset index in pandas
additional_df.reset_index()
df.reset_index()
print(additional_df)
print(df)

# write to csv file
with open('results/execution-time-1-computer.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(additional_df.columns.tolist())
    csvwriter.writerows(additional_df.values.tolist())
    csvwriter.writerow(df.columns.tolist())
    csvwriter.writerows(df.values.tolist())

# run statistics_graph.py file
# os.system("python3 postprocessing/statistics_graph.py")
