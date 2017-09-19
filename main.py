from SOM import SOM
from pcapParser import pcapParser
from pcapParserOld import pcapParserOld
from matplotlib import pyplot as plt
import plotly
import operator
import numpy as np
import calendar
from datetime import datetime


# Enter files here. Files found at: https://ll.mit.edu/ideval/data/1998data.html
tcpdump = r"outside.tcpdump"
tcpdump_list = r"tcpout.list"
#IMPORTANT: All pcap files use UTC time. List files times are based on the timezone where the data was recorded.
#The timezone_offset field is used to fix this. It is the hour offset from UTC time for the list file.
#The sample files sample_data01.tcpdump and tcpdump.list are 5 hours behind UTC. timezone_offset for these files should be 5.
#All other files are 4 hours behind UTC. timezone_offset for these files should be 4.
timezone_offset = 4



"""Runs the full nueral network.
   Since DDOS generates a ton of traffic at once, many of the files contain almost exclusively DDOS traffic.
   Due to the high amount of DDOS traffic, files will take a while to read in.
   For quick testing, use the sample file and lower the number of iterations/files."""
def main():


    #########################################
    # Parse the files.
    ##########################################
    pcapPar = pcapParser(tcpdump,tcpdump_list,timezone_offset=timezone_offset,max_packets=None,start_time=None,stop_time=None) #start and stop time are in unix time.
    tensors = pcapPar.tensors()
    labeledTensors = pcapPar.labeledTensors()

    ##########################################
    #Set up the SOM and train
    ##########################################
    som = SOM(n=10, m=10, dim=len(tensors[0]), n_iterations=2)
    som.train(tensors)
    som.label(labeledTensors)

    ##########################################
    # Get and print result
    ##########################################

    percentages = som.get_percentages()

    #Get accuracy matrix.
    # Note: Once the SOM is working, labeledTensors should belong to the test data opposed to the training data.
    matrix = som.get_accuracy_matrix(labeledTensors)
    print("attacks identified as attacks:"+str(matrix[0][0]))
    print("attacks identified as normal:"+str(matrix[0][1]))
    print("attacks identifed as other:"+str(matrix[0][2]))
    print("normal identified as attack:"+str(matrix[1][0]))
    print("normal identified as normal:"+str(matrix[1][1]))
    print("normal identified as other:"+str(matrix[1][2]))

    ###########################################
    # Plot
    ##########################################

    # Heat Map
    som.plot_heat_map()
    # Attack lines.
    som.plot_line_graph(labeledTensors,pcapPar.labler)

    plt.show()

def mainOld():
    """Old main function running files from our simulations."""

    # Normal and DDOS traffic ran over a long period of time.
    mixedLarge = "traffic-samples\mixed-traffic-sample\MixedDDOS"
    mixedLarge_1 = "traffic-samples\mixed-traffic-sample\MixedDDOS1"
    mixedLarge_2 = "traffic-samples\mixed-traffic-sample\MixedDDOS2"

    # Normal and DDOS traffic ran over a short period of time.
    mixedQuick = "traffic-samples\mixed-traffic-sample\MixedQuick"
    mixedQuick1 = "traffic-samples\mixed-traffic-sample\MixedQuick1"
    mixedQuick2 = "traffic-samples\mixed-traffic-sample\MixedQuick2"
    mixedQuick3 = "traffic-samples\mixed-traffic-sample\MixedQuick3"
    mixedQuick4 = "traffic-samples\mixed-traffic-sample\MixedQuick4"
    mixedQuick5 = "traffic-samples\mixed-traffic-sample\MixedQuick5"
    mixedQuick6 = "traffic-samples\mixed-traffic-sample\MixedQuick6"
    mixedQuick7 = "traffic-samples\mixed-traffic-sample\MixedQuick7"

    pcapPar = pcapParserOld(mixedQuick,mixedQuick1)

    print("File read in. Training.")
    pcap_tensors = pcapPar.pcap_list()

    # Set up the SOM
    som = SOM(n=5, m=5, dim=5, n_iterations=10)
    som.train(pcap_tensors)

    # Output simple color plot
    colors = som.color_inputsOld(pcapPar.pcap_dictionary())
    plt.imshow(colors)
    plt.title('Color SOM')
    plt.show()

if __name__ == "__main__":
    main()