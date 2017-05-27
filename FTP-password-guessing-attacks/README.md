# FTP password guessing attack neural network
A simple neural network designed to detect FTP Password Guessing attacks. The network uses a K-Means algorithm in order to categorize vsftpd log file connections as 
either attacks or non-attacks. On the sample log files, the algorithm performs well, correctly identifying most attacks and normal connections.

## Directory Layout

 
 - FakeFTPTrafficCreator : Simulates both normal ftp traffic as well as attacks. Has scripts to create new users on a Linux machine and then use those users to simulate traffic.
 
 - K-Means : Contains code to read in vsftpd log files and then categorize the ip-source of the connections as either attacks or normal traffic.  


