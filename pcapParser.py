from scapy.all import *
import netaddr
import sys


"""Contains functions designed to parse pcap files using scapy-python3: https://github.com/phaethon/scapy.
   Windows machines require winpcap to work with scapy: https://www.winpcap.org/"""
class pcapParser():

    """Parses the pcap file and stores it in self.pcap_array.
	   Args:
			pcap_file: A valid pcap file.
			pcap_files: Additional pcap files."""
    def __init__(self,pcap_file,*pcap_files):
        self.pcap_array= self.parseFile(pcap_file,*pcap_files)

    """Parses the pcap file.
       Args:
            pcap_files: Any number of valid pcap file names.
       Returns: An array of arrays. Each inner array represents a packet.
                Currenty, the array is structured as [source IP,protocol,packet type,length,time]
                where each field is a float."""
    def parseFile(self, *pcap_files):
        return_array = []
        for pcap_file in pcap_files:
            with PcapReader(pcap_file) as pcap_reader:
                for packet in pcap_reader:
                    packet_array = []
                    try:  # Things like ARP packets don't have a payload source.
                        packet_array.append(float(int(netaddr.IPAddress(packet.payload.src))))
                    except AttributeError:
                        packet_array.append(0.0)
                    packet_array.append(float(packet.proto))
                    packet_array.append(float(packet.pkttype))
                    try:  # Things like ARP packets don't have a packet length.
                        packet_array.append(float(packet.len))
                    except AttributeError:
                        packet_array.append(0.0)
                    packet_array.append(float(packet.time))
                    return_array.append(packet_array)
                print("%s file parsed" % (pcap_file))
        return return_array

    """Normalizes all values inside the array using the function 1/(1/+x).
        Args:
            array: An array of floats.
        Returns:
            The same array with all values normalized."""
    def normalizeData(self,array):
        for packet in array:
            packet[:] = [1/(1+x) for x in packet]
        return array


if __name__ == '__main__':
    if(len(sys.argv) <= 1):
        print("Usage: python pcapParser.py pcapfile1 pcapfile2 . . .")
    else:
        argvs = sys.argv
        argvs.pop(0)
        pcapPar = pcapParser(*argvs)
        normalized = pcapPar.normalizeData(pcapPar.pcap_array)