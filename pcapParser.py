from scapy.all import *

"""Contains functions designed to parse pcap files. Will need to be modified to match our needs later."""

"""Parses the pcap file.
   Args:
        pcap_file: A valid pcap file.
   Returns: An array of arrays. Each inner array represents a packet. Currenty, the array is structured as [src, dst,len,time]
            where each field is a string."""
def parse_file(pcap_file):
	return_array=[]
	packets = rdpcap(pcap_file)
	
	for packet in packets:
		packet_array=[]
		packet_array.append(packet.src)
		packet_array.append(packet.dst)
		packet_array.append(str(packet.len))
		packet_array.append(str(packet.time))
		return_array.append(packet_array)
	return return_array
		


if __name__ == '__main__':
    print parse_file("test_capture.pcap")
