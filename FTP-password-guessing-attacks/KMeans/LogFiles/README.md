# Log Files

 Sample log files used to initially test the FTPLogReader. Overfitting will likely be a huge problem for these log files. In the future, more robust log files will have to be created since these log files are rather small compared to those that would be generated in reality.
 
 Password guessing attacks were made from Kali Linux using hydra with the password list rockyou.txt. Due to time constraints, most attacks were stopped before they were successful. The exact command is:
  
~~~~

hydra -V 192.168.56.101 ftp -l test -P rockyou.txt

~~~~

where "user" was replaced by various users, both existant and nonexistant. Note that hydra will fail to tell you if the ip address is unreachable and will attempt to attack no matter what. Due to this, one must check that the ip address is reachable before the attack.

### Log Files
 
 - sample1.log : Simple sample where users randomly log in for a period of time and then log out. 30% of connections are attacks made by hydra.
 
 - sample2.log : More complex sample where some users are more active than others. 30% of connections are made by hydra.
 
 - sample3.log : Sample where only 10% of connections are attacks made by hydra.
 
 - sample4.log : Sample where 80% of connections are attacks made by hydra.
 
 - sample5.log : Larger version of sample2.log
 
 - attack.log : All attacks.
 
 - normal.log : All use data from ten users.
 
 
