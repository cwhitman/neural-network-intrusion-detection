from SOM import SOM
from pcapParser import pcapParser
from matplotlib import pyplot as plt

#Normal and DDOS traffic ran over a long period of time.
mixedLarge = "traffic-samples\mixed-traffic-sample\MixedDDOS"
mixedLarge_1 = "traffic-samples\mixed-traffic-sample\MixedDDOS1"
mixedLarge_2 = "traffic-samples\mixed-traffic-sample\MixedDDOS2"

#Normal and DDOS traffic ran over a short period of time.
mixedQuick = "traffic-samples\mixed-traffic-sample\MixedQuick"
mixedQuick1 = "traffic-samples\mixed-traffic-sample\MixedQuick1"
mixedQuick2 = "traffic-samples\mixed-traffic-sample\MixedQuick2"
mixedQuick3 = "traffic-samples\mixed-traffic-sample\MixedQuick3"
mixedQuick4 = "traffic-samples\mixed-traffic-sample\MixedQuick4"
mixedQuick5 = "traffic-samples\mixed-traffic-sample\MixedQuick5"
mixedQuick6 = "traffic-samples\mixed-traffic-sample\MixedQuick6"
mixedQuick7 = "traffic-samples\mixed-traffic-sample\MixedQuick7"

"""Runs the full nueral network.
   Since DDOS generates a ton of traffic at once, many of the files contain almost exclusively DDOS traffic.
   Due to the high amount of DDOS traffic, files will take a while to read in.
   For quick testing, use the mixedQuick files and lower the number of iterations/files."""
def main():
    #Parse the files.
    pcapPar = pcapParser(mixedQuick,mixedQuick1,mixedQuick2,mixedQuick3,mixedQuick5,mixedQuick6,mixedQuick7)

    print("File read in. Training.")
    pcap_tensors = pcapPar.pcap_list()


    #Set up the SOM
    som = SOM(n=5, m=5, dim=5, n_iterations=10)
    som.train(pcap_tensors)

    # Output simple color plot
    colors = som.color_inputs(pcapPar.pcap_dictionary())
    print(pcapPar.pcap_dictionary())
    plt.imshow(colors)
    plt.title('Color SOM')
    plt.show()

if __name__ == "__main__":
    main()