import re
import os.path
import logging
import datetime
import netaddr
from time import ctime
from functools import reduce


__author__ = "Caleb Whitman"
__version__ = "1.0.0"
__email__ = "calebrwhitman@gmail.com"

"""Parses a vsftpd logfile and outputs a tensor representing the file.
   In order to use the FTPLogReader, instantiate a new class and then call getConnectionTensors to get the tensors."""
class FTPLogReader:

    """ Instanitates a new FTPLogReader.
        Args:
            file (string): The vsftpd logfile to read from.
            position (int): The position to start reading the file. Defaults to start of file.
            logFile (string): The logfile that any errors will be logged to. Defaults to FTPParseLog.log"""
    def __init__(self,file,position =0,logFile="FTPParseLog.log" ):
        if( not os.path.isfile(file)):
            raise IOError("File not found.")
        logging.basicConfig(filename=logFile, level=logging.WARNING)
        logging.getLogger().addHandler(logging.StreamHandler())
        self.file =file
        self.position = position

    """ Returns tensors from the logfile representing the connection information.
         Returns:
            [ [5] ]: A list of connections. [[ip,average_datetime_difference,ok_login_average,connect_average,fail_login_average ]]"""
    def getConnectionTensors(self):
        dict = self.__parseLogFile__()
        conn_dict = self.__getConnections__(dict)
        ips = self.__sortByIP__(conn_dict)
        connections = self.__combineConnections__(ips)

        return connections

    #Converts the given ip into an interger.
    def turnIptoInt(self,string_ip):
        return float(int(netaddr.IPAddress(string_ip)))

    """Takes a list of ips and combines all connections within that IP list.
       Returns: [[ip,average_datetime_difference,ok_login_average,fail_login_average,total_connections ]]"""
    def __combineConnections__(self,ips):

        result_list=[]
        max_average_time_difference =0
        max_total_connections = 0
        for ip in ips:
            connection=[]
            datetimes=[]

            #getting connection values
            ok_login_num=0
            connect_num = 0
            fail_login_num = 0
            total_connections = 0
            for dict in ips[ip]:
                if(dict["status"] == "OK LOGIN"):
                    ok_login_num+=1
                if(dict["status"] == "CONNECT"):
                    connect_num+=1
                if(dict["status"] == "FAIL LOGIN"):
                    fail_login_num+=1
                datetimes.append(unix_time_seconds(dict["datetime"]))
                total_connections+=1

            #appending everything to connection
            connection.append(self.turnIptoInt(ip))
            datetime_diffs = []
            datetimes.sort()
            for i in range(1,len(datetimes)):
                datetime_diffs.append(datetimes[i]-datetimes[i-1])
            if(len(datetime_diffs)>0):
                average_time_difference = reduce(lambda x, y: x + y, datetime_diffs) / float(len(datetime_diffs))
            else:
                average_time_difference = 800
            if(average_time_difference > max_average_time_difference):
                max_average_time_difference=average_time_difference
            connection.append(average_time_difference)
            connection.append(ok_login_num/total_connections)
            #connection.append(connect_num/total_connections)
            connection.append(fail_login_num/total_connections)
            connection.append(total_connections)
            if(total_connections>max_total_connections):
                max_total_connections=total_connections

            result_list.append(connection)

        #Normalize average distance with total number of connections.
        normalize = max_average_time_difference/max_total_connections
        print(normalize)
        for connection in result_list:
            connection[1] = connection[1]/normalize
        return result_list


    """Parses the logfile and returns a list of dictionaries representing every line.
        Returns:
            [{}]: A list of dictionaries representing the logfile."""
    def __parseLogFile__(self):

        #Open file and go to the correct staring position.
        fp = open(self.file, 'r')
        fp.seek(self.position,0)
        result = []

        #Read and parse each line.
        for line in fp:
            try:
                result.append(self.__parseLine__(line))
            except CantParseException:
                logging.warning("%s: Line unable to be parsed: %s"%(ctime(),line))

        #Get end of file position
        self.position=fp.tell()
        #Close file pointer
        fp.close()
        return result

    """ Looks through the dictionaries and filters out any that do not represent a connection.
         Args:
                dictinaries ([{}]): The list of dictionaries representing the ftp log.
        Returns:
            [{}]: A list of dictionaries representing connections and connection attempts.
            """
    def __getConnections__(self,dictionaries):

        result=[]
        for dict in dictionaries:
            if(dict["status"]=="CONNECT" or dict["status"]=="OK LOGIN" or dict["status"]=="FAIL LOGIN"):
                result.append(dict)

        return result



    """ Sorts the list of dictionaries by IP addresses.
         Args:
                dictinaries ([{}]): The list of dictionaries representing the ftp log.
        Returns:
            {ip:[{}]}: A dictionary holding each list of dictionaries for a given ip.
            """
    def __sortByIP__(self,dictionaries):
        result = {}
        for dict in dictionaries:
            if dict["ip"] in result:
                result[dict["ip"]].append(dict)
            else:
                result[dict["ip"]]=[dict]
        return result

    """ Parses the line and returns a dictionary representing that line.
        Returns:
            {}: A dictionary in the format {datetime,pid,username,status,ip,parameters}
            -datetime is a datetime object representing the time the action occured.
            -pid is the process id on the server.
            -username is the user name that carried out the connection. May be None.
            -status is the status of the connection.
            -ip is the ip address of the client.
            -parameters is a string containing a comma seperated list of parameters. May be None.
            """
    def __parseLine__(self,line):
        logPattern = r"" \
                     r"([^ ]*) " \
                     r"([^ ]*) " \
                     r"([^ ]*) " \
                     r"([^ ]*) " \
                     r"([^ ]*) " \
                     r"(\[[^\[]*\]) " \
                     r"(\[[^\[]*\] )?" \
                     r"([^:]*): " \
                     r"Client \"([^\"]*)\"" \
                     r",? ?(.+)?"
        compiledPattern = re.compile(logPattern)
        matched = compiledPattern.match(line)
        if (matched is None):
            raise CantParseException
        else:
            #Do further parsing on any of the components and then put the result into the dictionary
            groups = matched.groups()
            months = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
            time = groups[3].split(":")
            date = datetime.datetime(int(groups[4]), months[groups[1]], int(groups[2]), int(time[0]), int(time[1]), int(time[2]))
            if groups[6] is not None:
                username = groups[6].replace("[","").replace("]","")
            else:
                username=None

            resultDictionary = {"datetime": date, "pid": groups[5], "username": username, "status": groups[7],
                                "ip": groups[8], "parameters": groups[9]}

        return resultDictionary

    """Parses the parameters into a dictionary. Each dictionary will hold a different value depending on the parameter format.
        Parameters that are not reconized are not returned.
        Currently only parses out the the parameter "PORT 123,123 etc...."
        Args:
            params (string): A comma seperated listed of paraemters
     Param: params, the parameters in a comma seperated string format."""
    def __parseParameter__(params):
        if (params is None):
            return
        returnDictionary = {}
        # Ports
        portsParam = r"\"PORT ([0-9]*,?)*\""
        compiledPattern = re.compile(portsParam)
        matched = compiledPattern.match(params)
        if (matched is not None):
            group = matched.group(0)
            portRemove = group.split()
            ports = portRemove[1].split(",")
            portNums = []
            [portNums.append(int(x.replace("\"", ""))) for x in ports[1:]]
            returnDictionary["PORT"] = portNums

        return returnDictionary


"""Gets the time in seconds from the given datetime."""
def unix_time_seconds(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds()

"""Thrown when we read in a line we can't parse."""
class CantParseException(Exception):
    pass
