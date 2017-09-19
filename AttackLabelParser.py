import re
import time
from datetime import datetime
import calendar


class AttackLabels():
    """
    Parses the tcpdump.list file.
    Methods isAttack and attackName can them be used to get the attack and it's name.
    """

    def __init__(self,file_path,timezone_offset):
        """
        Reads in the file and creates an object to remember and handle the attacks.
        Args:
            file_path: string represent a valid tcpdump.list file
            timezone_offset: Second offset of timezone from UTC.
        """

        # How much error should be expected when comparing the tcpdump file to the tcpdump.list file in seconds.
        self.timeError = 1

        self.attacks=[]

        self.readFile(file_path,timezone_offset)


    def readFile(self,file_path,timezone_offset):
        """Reads in a new file and adds it to the current labler.
           Arges:
                 file_path: string represent a valid tcpdump.list file
                 timezone_offset: Second offset of timezone from UTC."""

        # line number, start date, start time, duration, service, source port, destination port, source ip, destination ip, attack score, names
        file_regex = re.compile("(.+) (.+ .+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)")

        with open(file_path,"r") as file:
            for line in file:

                attributes= re.match(file_regex, line).groups()
                start_time = calendar.timegm(datetime.strptime(attributes[1], "%m/%d/%Y %H:%M:%S").utctimetuple())+timezone_offset*3600
                stripped_time = datetime.strptime(attributes[2], "%H:%M:%S")
                stop_time = stripped_time.hour*3600 + stripped_time.minute*60+stripped_time.second+start_time
                service = attributes[3]
                sport = attributes[4]
                dport = attributes[5]
                src_ip = attributes[6]
                dest_ip = attributes[7]
                attack = True if attributes[8]=="1" else False
                attack_names = attributes[9].split(",")

                if attack:
                    self.attacks.append([start_time,stop_time,sport,dport,src_ip,dest_ip,attack_names])

    def isAttack(self,src_ip,dest_ip,sport,dport,time):
        """Returns True if the input information indicates an attack.
           Args:
                src_ip: source ip
                dest_ip: destination ip
                sport: source port
                dport: destination port
                time: time of traffic.
            Returns: True if attack occured around the time with the corresponding information."""
        #print("%s %s %s %s %s"%(src_ip,dest_ip,sport,dport,time))
        attacks = [x for x in self.attacks if
                   x[0]<=time+self.timeError and
                   time -self.timeError <=x[1] and
                   str(sport) == x[2] and
                   str(dport) == x[3] and
                   x[4]==src_ip and
                   x[5]==dest_ip
                   ]
        # if len(attacks)>0:
        #     print(attacks,":",sport,":",dport)
        return len(attacks)>0

    def attackNames(self,src_ip,dest_ip,sport,dport,time):
        """Returns names of attacks with the given information around the time.
           Args:
                src_ip: source ip
                dest_ip: destination ip
                sport: source port
                dport: destination port
                time: time of traffic.
            Returns: A list of attack names or an empty list if there were none."""
        return [attack for x in self.attacks if
                   x[0]<=time+self.timeError and
                   time -self.timeError <=x[1] and
                   str(sport) == x[2] and
                   str(dport) == x[3] and
                   x[4]==src_ip and
                   x[5]==dest_ip
                   for attack in x[-1]]

if __name__ == '__main__':
    labler = AttackLabels("tcpdump.list")
    labler.isAttack("192.168.1.40","192.168.0.20",885600233.0)
