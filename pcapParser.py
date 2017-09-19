from scapy.all import *
import netaddr
import sys
from AttackLabelParser import AttackLabels
from datetime import datetime

class pcapParser():
    """Contains functions designed to parse pcap files using scapy-python3: https://github.com/phaethon/scapy.
       Windows machines require winpcap to work with scapy: https://www.winpcap.org/

       After initialization, parsed, normalized pcap file can be accessed through the method tensors().
       """

    def __init__(self,pcap_file,label_file,timezone_offset,max_packets=None,start_time=None,stop_time=None):
        """Parses the pcap file and stores it in self.pcap_array.
           Args:
                pcap_file: A valid pcap file.
                label_file: The tcpdump.list file containing attack labels.
                timezone_offset: Second offset of timezone from UTC of the label file.
                max_packets: Maximum number of packets to read in.
                start_time: UTC time in seconds to start reading in packets.
                stop_time: UTC time in seconds to stop reading in packets.
                """
        self.__ip_dictionary__ = {}
        self.__tensors__ = []
        self.labler = AttackLabels(label_file,timezone_offset)
        self.addFileToTensor(pcap_file,max_packets=max_packets,start_time=start_time,stop_time=stop_time)

    def tensors(self):
        """Returns [[ exponential moving average of time between packets,
                     total number of packets received in last second,
                     protocol type,
                     bytes sent,
                     average Time between connection for this IP,
                     total number of connections for this IP]
           Return time is order n as the method normalizes the data before returning it.
           """
        return [x[0] for x in self.__normalizeData__(self.__tensors__)]

    def labeledTensors(self):
        """Returns [
                    [
                     exponential moving average of time between packets,
                     total number of packets received in last second,
                     protocol type,
                     bytes sent,
                     average Time between connection for this IP,
                     total number of connections for this IP],
                    is_attack,
                    attack name,
                    packet time
                    ]
           Return time is order n as the method normalizes the data before returning it.
        """
        return self.__normalizeData__(self.__tensors__)

    def addFileToTensor(self, pcap_file,label_file=None,max_packets=None,start_time=None,stop_time=None):
        """Adds the pcap file to the tensor.
           Args:
                pcap_file: A valid pcap file.
                label_file: The tcpdump.list file containing attack labels.
                max_packets: Maximum number of packets to read in.
                start_time: Unix time (# seconds after epoch) to start reading packets.
                end_time: Unix time (# seconds after epoch) to stop reading packets.
            """


        if label_file is not None:
            labler.readFile(label_file)

        with PcapReader(pcap_file) as pcap_reader:
            last_time = -1
            exponential_average=0
            time_tensor = []
            for packet in pcap_reader:

                # Check constraints to see when to start and stop reading in packets.
                if max_packets is not None:
                    max_packets -= 1
                    if max_packets < 0:
                        break
                if start_time is not None:
                    if packet.time < start_time:
                        continue
                if stop_time is not None:
                    if packet.time > stop_time:
                        break


                #Get fields.
                #Some packets, such as ARP, are missing for most of the fields we are training on.

                #Source IP
                try:
                    src = packet.payload.src
                except AttributeError:
                    src = ""

                #Destination IP
                try:
                    dst = packet.payload.dst
                except AttributeError:
                    dst = ""

                #Source port
                try:
                    sport = packet.payload.sport
                except:
                    sport = ""

                #Destination port
                try:
                    dport = packet.payload.dport
                except:
                    dport=""

                # Add ip to dictionary
                if(src not in self.__ip_dictionary__):
                    self.__ip_dictionary__[src] = [0.0, 0.0, 0.0]
                    #[Time of last connection for ip,
                    #average Time between connection for this IP,
                    #total number of connections for this IP]

                #Update time difference
                if last_time != -1:
                    time_diff = packet.time - last_time
                else:
                    time_diff=0

                #Data for that tensors IP
                ip_data = self.__ip_dictionary__[src]

                #time between packets
                time_moving_average = self.exponentialMovingAverage(time_diff,exponential_average,0.1)

                #Packets in last second.
                time_tensor.append(packet.time)
                new_time_tensor = [ x for x in time_tensor if packet.time-x <1]
                time_tensor = new_time_tensor
                packets_last_second = len(time_tensor)

                #Packet prototype
                try:
                    proto=packet.proto
                except AttributeError:
                    proto=0.0


                #Payload length
                try:
                    payload_len = float(packet.payload.len)
                except AttributeError:
                    payload_len = 0.0

                #Total number of connections for this IP
                ip_data[2]+=1
                total_ip_connections = ip_data[2]

                #Average time between connections for this IP
                ip_time_diff = packet.time - ip_data[0] #average time between connections
                ip_data[1] = (ip_data[1]*ip_data[2] + ip_time_diff)/ip_data[2]
                average_ip_time = ip_data[1]

                # time of last connection (IP data only. Not added to tensor.)
                ip_data[0]=packet.time
                last_time = packet.time

                #new tensor
                tensor=[]
                tensor.append(time_moving_average)
                tensor.append(packets_last_second)
                tensor.append(float(proto))
                tensor.append(payload_len)
                tensor.append(total_ip_connections)
                tensor.append(average_ip_time)

                #Attack
                attack = self.labler.isAttack(src,dst,sport,dport,packet.time)
                attack_names = self.labler.attackNames(src,dst,str(sport),str(dport),packet.time)
                self.__tensors__.append([tensor,attack,attack_names,packet.time])

        return self.__tensors__

    def __normalizeData__(self,tensors = None):
        """Normalizes all values inside the tensor using the function 1/(1/+x).
            Args:
                tensors: tensors produced using addFileToDictionary. Defaults to self.__tensors__.
            Returns:
                The same tensors with all values normalized.
                """
        if(tensors is None):
            tensors = self.__tensors__
        new_tensors=[]
        for tensor in tensors:
             new_tensors.append( [ [1/(1+x) for x in tensor[0] ] , tensor[1],tensor[2],tensor[3] ])
        return new_tensors

    def exponentialMovingAverage(self,new,old, alpha):
        """Returns the exponenatial moving average.
           Params:
                 new: new valuve to be added to the mean
                 old: old mean
                 alpha: value from 0 to 1 determining how quickly the old mean decays. A value of 1 completely forgets the old mean.
            Returns: the new exponenatial moving average.
            """
        return alpha * new + (1 - alpha)* old

if __name__ == '__main__':
    if(len(sys.argv) <= 1):
        print("Usage: python pcapParser.py tcpdumpFile tcpdumpList")
    else:
        argvs = sys.argv
        argvs.pop(0)
        pcapPar = pcapParser(argvs[0],argvs[1])