from SOM import SOM
from pcapParser import pcapParser
from matplotlib import pyplot as plt

normal = "traffic-samples\\normal-traffic-sample\TCP.pcap"
attackhping3 = "traffic-samples\\attack-traffic-sample\hping3Slow.pcap"
attackLOIC = "traffic-samples\\attack-traffic-sample\LOICSlow.pcap"

"""Runs the full nueral network."""
def main():
    #Parse the files.
    pcapPar = pcapParser(normal,attackhping3,attackLOIC)
    pcap_tensors = pcapPar.pcap_list()

    #Set up the SOM
    som = SOM(n=100, m=100, dim=5, n_iterations=100)
    som.train(pcap_tensors)

    # Output simple color plot
    colors = som.color_inputs(pcapPar.pcap_dictionary())
    plt.imshow(colors)
    plt.title('Color SOM')
    plt.show()

if __name__ == "__main__":
    main()