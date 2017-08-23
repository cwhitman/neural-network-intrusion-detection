from SOM import SOM
from pcapParser import pcapParser
from pcapParserOld import pcapParserOld
from matplotlib import pyplot as plt

# Enter files here. Files found at: https://ll.mit.edu/ideval/data/1998data.html
tcpdump = r"sample_data01.tcpdump"
tcpdump_list = r"E:\tcpdump.list"


"""Runs the full nueral network.
   Since DDOS generates a ton of traffic at once, many of the files contain almost exclusively DDOS traffic.
   Due to the high amount of DDOS traffic, files will take a while to read in.
   For quick testing, use the mixedQuick files and lower the number of iterations/files."""
def main():

    #Parse the files.
    pcapPar = pcapParser(tcpdump,tcpdump_list)
    tensors = pcapPar.tensors()

    #Set up the SOM
    som = SOM(n=5, m=5, dim=len(tensors[0]), n_iterations=1)
    som.train(tensors)

    # Get output grid
    image_grid = som.color_map(pcapPar.labeledTensors())

    #Get accuracy matrix.
    percentages = som.percentageIdentification(pcapPar.labeledTensors())
    matrix = som.getAccuracyMatrix(pcapPar.labeledTensors(),percentages)
    print("attacks identified as attacks:"+str(matrix[0][0]))
    print("attacks identified as normal:"+str(matrix[0][1]))
    print("attacks identifed as other:"+str(matrix[0][2]))
    print("normal identified as attack:"+str(matrix[1][0]))
    print("normal identified as normal:"+str(matrix[1][1]))
    print("normal identified as other:"+str(matrix[1][2]))

    # Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    plt.show()


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