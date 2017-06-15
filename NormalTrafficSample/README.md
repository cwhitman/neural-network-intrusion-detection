# Sample Normal Traffic

Simulated normal traffic. Traffic was simulated using [Distributed Internet Traffic Generator](http://www.grid.unina.it/software/ITG/)(D-ITG).
For now, only simple single traffic simulations were carried out one at a time. In the future, simultaneous, advanced simulations can be carried out at once using
D-ITG's script function.
Traffic was captured using tcpdump with the command `tcpdump -n -nn -i any -w capture_file'.

## File Description

All files do not contain payloads, put the distribution and size of the packets approximately matches real life traffic.

 - Csa.pcap: Simulated Counter Strike traffic for an active player for 15 minutes. Source IP is 198.168.56.102. Receiving IP is 198.168.56.101.  Generated using the command 
 './ITGSend Csa -a 192.168.56.101 -t 900000'.
 
 - Telnet.pcap: Similated Telnet traffic for 15 minutes. Source IP is 198.168.56.102. Receiving IP is 198.168.56.101. Generated using the command 
 ''./ITGSend Telnet -a 192.168.56.101 -t 900000'.
 
 - TCP.pcap: Similated TCP traffic for 15 minutes. Traffic distribution and packet size follow a normal distribution. Source IP is 198.168.56.102. Receiving IP is 198.168.56.101. Generated using the command 
 ''./ITGSend -T TCP -a 192.168.56.101 -N 1000 200 -n 512 50 -t 900000'.




