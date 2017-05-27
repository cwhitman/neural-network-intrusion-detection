import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

from functions import *

__author__ = "Caleb Whitman"
__version__ = "1.0.0"
__contributors__ = ["Caleb Whitman", "Brant Dolling", "Jacob Sanderlin"]
__email__ = "calebrwhitman@gmail.com"


n_features = 2
n_clusters = 2
n_samples_per_cluster = 2
seed = 700
embiggen_factor = 70
max_iterations = 100
stop_threshold = 1

#Create variables/samples
samples = get_FTP_tensors("LogFiles/sample1.log")

#real_centroids,samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

#Initialize variables in tensorflow
model = tf.global_variables_initializer()

#Run tensorflow
with tf.Session() as session:
    #Run the initial set up
    sample_values = session.run(samples)
    # Run the update.
    sum=0
    for _ in range(max_iterations):

        updated_centroid_value,connection_groups = session.run(updated_centroids)

        stop,sum = should_stop(connection_groups, updated_centroid_value,stop_threshold,sum)
        #Stopping condition.
        if stop:
            print("Threshold or Convergance Reached.")
            break

        #updating values
        nearest_indices = assign_to_nearest(samples,tf.constant(updated_centroid_value))
        updated_centroids = update_centroids(samples, nearest_indices, n_clusters)


#Determining the accuracy.
ipFileReader = open("ips.txt", "r")
ipString = ipFileReader.readline()
goodIPs=[]
badIPs=[]

while True:
    ipString = ipFileReader.readline().replace("\n","")
    if(ipString == "attack ips"):
        while True:
            ipString = ipFileReader.readline().replace("\n","")
            if ipString == '':
                break
            badIPs.append(turnIptoInt(ipString))
        break
    goodIPs.append(turnIptoInt(ipString))


#Printing out the accuracy and plotting
good_percentage, bad_percentage = getGoodBadIPCount(connection_groups,0, goodIPs, badIPs)
print("Group 1 has %.5s percent of the good IPs and has %.5s percent of the bad IPs."%(good_percentage,bad_percentage))

good_percentage, bad_percentage = getGoodBadIPCount(connection_groups,1, goodIPs, badIPs)
print("Group 2 has %.5s percent of the good IPs and has %.5s percent of the bad IPs."%(good_percentage,bad_percentage))

#plot
plot_clusters(sample_values, updated_centroid_value,goodIPs,badIPs)
