# Neural Network Intrusion Detection System

This project is an attempt to create a neural network intrusion detection system against FTP password guessing attacks, DOS attacks, and DDOS attacks.
Currently, a simple neural network using the [K-Means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering) has been designed to detect FTP password guessing attacks.
The network correctly identifies around 80% of normal connections and 95% of attacks for the test cases.

## Organization

 All TODOs for the project are listed under [issues](https://github.com/cwhitman/neural-network-intrusion-detection/issues). Any intermediate research that is made will be posted under the relevant issue.
 
 Once implemented, the total neural network system will be organized into three pieces:
 
   1. Data Input Formatting: Reads in the data and formats it into a tensor that the network can use. (Complete)
   
   2. Neural Network: The neural network itself. Uses self-organizing maps in order to separate attacks from non-attacks. (In Progress)
    
       - Train: Train the neural network. Structures the centroids so that they match the approximate distribution of the input data.

       - Identification: Identify which parts of the map correspond to attack traffic and which parts correspond to normal traffic.

       - Test: Run test data on the network and see what percentage of attacks/normal traffic is correctly identified. 
   
   3. Data Output Formatting: Format the data from the neural network into a human readable format. (In Progress)
