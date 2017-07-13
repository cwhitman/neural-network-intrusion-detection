from scapy.all import *
import netaddr
import sys


"""Contains functions designed to parse pcap files using scapy-python3: https://github.com/phaethon/scapy.
   Windows machines require winpcap to work with scapy: https://www.winpcap.org/

   After initialization, parsed, normalized pcap file can be accessed through the method pcap_dictionary().
   """
class pcapParser():

    """Parses the pcap file and stores it in self.pcap_array.
	   Args:
			pcap_file: A valid pcap file.
			pcap_files: Additional pcap files."""
    def __init__(self,pcap_file,*pcap_files):
        self.__pcap_dictionary__ = {}
        self.__pcap_dictionary_data__ = {}
        self.addFileToDictionary(pcap_file,*pcap_files)

    """Returns the normalized pcap file in dictionary form.
       If the stored pcap dictionary is large, this method may take a while to return as it normalizes the data before returning it.
       Returns: { ip , [Time of last connection, Total number of connections,Average Time between connection,
       Total number of connections last hour, Total number of connections this month] }"""
    def pcap_dictionary(self):
        return self.__normalizeData__()

    """Returns the normalized pcap file in list form.
       If the stored pcap list is large, this method may take a while to return as it normalizes the data before returning it.
       Returns: [Time of last connection, Total number of connections,Average Time between connection,
       Total number of connections last hour, Total number of connections this month]"""
    def pcap_list(self):
        dict = self.pcap_dictionary()
        list = []
        for key in dict:
            list.append(dict[key])
        return list

    """Updates the connections last hour and connections this month values in the __pcap_dictionary__.
        Checks to see if it has been an hour and/or month since that value was last updated and then resets the
        values appropriately.
       Args:
            time: The current time as seconds since the epoc.
        Retuns: self.__pcap_dictionary__"""
    def updateConnectionTimes(self,time):
        for key in self.__pcap_dictionary__:
            if __name__ == '__main__':
                if(time - self.__pcap_dictionary_data__[key][0] > 3600): #total number of connections last hour
                    self.__pcap_dictionary_data__[key][0]=time
                    self.__pcap_dictionary__[key][3] = 0
                if (time - self.__pcap_dictionary_data__[key][1] > 3600):  # total number of connections last month
                    self.__pcap_dictionary_data__[key][1] = time
                    self.__pcap_dictionary__[key][4] = 0

    """Adds the pcap files to self.__pcap_dictionary__.
       self.__pcap_dictionary__ is a dictionary where key is the source IP and value is
       the tuple, [Time of last connection, Total number of connections,Average Time between connection,
       Total number of connections last hour, Total number of connections this month].
        self.__pcap_dictionary_data__ is used to keep track of things such as hour_last_reset for each entry.
        It is organized as { ip, [hour_last_reset,month_last_reset] }
       Args:
            pcap_files: Any number of valid pcap file names.
        Retuns: self.__pcap_dictionary__"""
    def addFileToDictionary(self,*pcap_files):
        for pcap_file in pcap_files:
            with PcapReader(pcap_file) as pcap_reader:
                last_time = -1
                for packet in pcap_reader:
                    try: #Some packets, such as ARP, don't contain a source IP address.
                        src = packet.payload.src
                    except AttributeError:
                        src = "No IP"

                    if(src not in self.__pcap_dictionary__): #Add ip to dictionary
                        self.__pcap_dictionary__[src] = [packet.time, 0.0, 0.0, 0.0, 0.0]
                        self.__pcap_dictionary_data__[src] = [float(packet.time), float(packet.time)]

                    list = self.__pcap_dictionary__[src]
                    list_data = self.__pcap_dictionary_data__[src]

                    list[1]+=1 #total number of connections
                    time_diff = packet.time - list[0] #average time between connections
                    list[2] = (list[2]*list[1] + time_diff)/list[1]
                    list[0]=packet.time #time of last connection
                    last_time = packet.time
                    if(packet.time - list_data[0] > 3600): #total number of connections last hour
                        list_data[0] = packet.time
                        list[3] = 0
                    else:
                        list[3]+=1
                    if(packet.time - list_data[1] > 2592000):#total number of connetions last month
                        list_data[1] = packet.time
                        list[4] = 0
                    else:
                        list[4] += 1

                #Update the connections last hour and connections last month field
                #based on the time that the last packet arrived.
                if (last_time != -1):
                    self.updateConnectionTimes(last_time)
        return self.__pcap_dictionary__



    """Normalizes all values inside the dictionary using the function 1/(1/+x).
        Args:
            dictionary: dictionary produced using addFileToDictionary. Defaults to self.__pcap_dictionary__.
        Returns:
            The same dictionary with all values normalized."""
    def __normalizeData__(self,dictionary = None):
        if(dictionary is None):
            dictionary = self.__pcap_dictionary__
        new_dictionary={}
        for key in dictionary:
             new_dictionary[key]= [1/(1+x) for x in dictionary[key]]
        return new_dictionary

    """Un-normalizes all values inside the dictionary assuming that all values were perviously normalized using 1/(1/+x).
            Args:
                dictionary: dictionary produced using addFileToDictionary. Defaults to self.__pcap_dictionary__.
            Returns:
                The same dictionary with all values un-normalized."""
    def __unNormalizeData__(self, dictionary=None):
        if (dictionary is None):
            dictionary = self.__pcap_dictionary__
        new_dictionary = {}
        for key in dictionary:
             new_dictionary[key]= [1/x -1 for x in dictionary[key]]
        return new_dictionary


    """Parses the pcap file into a simple array as opposed to a dictionary.
       May be removed in the future. Exists right now as an example.
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


if __name__ == '__main__':
    if(len(sys.argv) <= 1):
        print("Usage: python pcapParser.py pcapfile1 pcapfile2 . . .")
    else:
        argvs = sys.argv
        argvs.pop(0)
        pcapPar = pcapParser(*argvs)

        print(pcapPar.pcap_dictionary())