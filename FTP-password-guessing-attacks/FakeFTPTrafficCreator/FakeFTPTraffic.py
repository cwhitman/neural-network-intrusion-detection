#Generates fake FTP traffic from a given user list.
#First, reads in a user list and assigns each user 1 IP address.
#Next Assigns 20 IP addresses for attacks.
#All IP addresses are written out to a file for later access.
#Finally, alternates between making legidiment connections using the user IP
#addresses and attacking the server using the attack IP addresses with hydra.
#During a given session, not all users and IP addresses may be used.
#Runs forever until interuppted with cntl C

import os
import random
import ftplib
from ftplib import FTP
from time import sleep
import subprocess
import socket, struct, fcntl

__author__ = "Caleb Whitman"
__version__ = "1.0.0"
__email__ = "calebrwhitman@gmail.com"

"""Sets the ip address of the interface.
   Code retrieved from : http://stackoverflow.com/questions/20420937/how-to-assign-ip-address-to-interface-in-python
   Args:
        iface (string): intreface to assign ip
        ip    (string): new ip address."""
def setIpAddr(iface, ip):
    SIOCSIFADDR = 0x8916
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bin_ip = socket.inet_aton(ip)
    ifreq = struct.pack('16sH2s4s8s', iface, socket.AF_INET, '\x00' * 2, bin_ip, '\x00' * 8)
    fcntl.ioctl(sock, SIOCSIFADDR, ifreq)

"""Returns the difference between two lists"""
def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

"""Changes the ip address to the users ip address. Logs in, waits for some time, and then logs out.
   Around 1/10th of log in attempts will log in with the wrong password.
   Around 1/10 of the log in attempts will be anonymous
   Args:
        users [(string),(string),(string)] : [username, password,ip]"""
def LogIn(host,user):
    setIpAddr('eth0', user[2])  # Change the ip address the user is on
    ftp = FTP(host)  # connect to host, default port
    if(random.random()<0.1):
        ftp.login()  # user anonymous, passwd anonymous
    else:
        try:
            password=user[1]
            if(random.random()<0.1): #See if user "forgets" their password
                password=user[1]+"1"
            ftp.login(user[0],password)
        except ftplib.error_perm: #Don't care if we have a fail log in attempt
            pass
    sleep(random.randint(user[3][0], user[3][1])) #User does "stuff"
    try:
        ftp.quit() #Will fail if the user does not log in. Don't care.
    except Exception:
        pass

"""Uses hyrdra to attack the host for around 20 seconds.
   Args:
        host (string): ip address of host
        usernames [(string)]: list of possible usernames.
        ip (string): attackers ip"""
def attack(host,user,ip):
    setIpAddr('eth0', ip)  # Change the ip address the user is on
    subprocess.call(["timeout",20,"hydra","-V",host,"ftp","-l",user.strip('\n\t\r '),"-P", "rockyou.txt"])

"""Assuming the last element in choices is a weight, chooses an item in choices bases on weight."""
def weighted_choice(choices):
   total = sum(c[-1] for c in choices)
   r = random.uniform(0, total)
   upto = 0
   for c in choices:
      if upto + c[-1] >= r:
         return c
      upto += c[-1]

def main():
    users=[]
    users_file = "users.txt"
    ip_file="ips.txt"
    base_ip = "192.168.56."
    host_ip_end = 101;
    with open(users_file) as f:
        for line in f.readlines(0):
            namePass = line.split(';')
            user = namePass[0].strip('\n')
            password = namePass[1].strip('\n')
            users.append([user,password])

    #loading the ips. Assumes that ip file was written in a very specific format.
    attack_ips=[]
    with open("ips.txt","r") as f:
        for i in range(0,len(users)):
            line = f.readline().split("\n")
            users[i].append(line)
        f.readline();#skipping attack_ip line
        for line in f:
            attack_ips.add(line.split("\n"))



    """
    #Generate ips
    ips=[]
    for i in range(1,len(users)+22):
        if(i==host_ip_end): continue #host ip
        ips.append(base_ip+str(i))
    #Give each user a random ip

    user_ips = random.sample(ips,len(users))
    for i in range(0,len(user_ips)):
        users[i].append(user_ips[i])
    attack_ips = diff(ips,user_ips)

    #Write the ips to a file.
    with open(ip_file,'w') as f:
        f.write("user ips\n")
        for ip in user_ips:
            f.write(ip+"\n")
        f.write("attack ips\n")
        for ip in attack_ips:
            f.write(ip+"\n")
            """

    #Give each user a usage weight.
    #And a time interval for how long they log in.
    for user in users:
        lower_usage = random.randint(1,3)
        upper_usage = random.randint(4,10)
        user.append((lower_usage,upper_usage))
        weight = random.randint(1, 3)
        user.append(weight)




    #Running the fake traffic
    with open("usernames.txt") as f:
        usernames = f.readlines()

    while(True):
        if(random.random()>0.3):
            user= weighted_choice(users) #get a random user to log in
            print("normal log in with "+user[0])
            LogIn(base_ip+str(host_ip_end),user) #log in
        else:
            print("Attacking!")
            attack(base_ip+str(host_ip_end),random.choice(usernames),random.choice(attack_ips))




if __name__ == '__main__':
    main()