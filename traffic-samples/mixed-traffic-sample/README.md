# Sample Mixed Traffic

Simulated mixed traffic. Normal traffic was simulated using [Distributed Internet Traffic Generator](http://www.grid.unina.it/software/ITG/)(D-ITG).
DOS and DDOS traffic was generated using [hping3](https://tools.kali.org/information-gathering/hping3).
Traffic was captured using tcpdump with the command `tcpdump -i any -w capture_file -C 100`.
Server IP address is 10.0.0.1. 
Attack traffic has ip addresses ranging from 10.0.0.128 to 10.0.0.254 inclusive.
Normal traffic has ip addresses ranging from 10.0.0.2 to 10.0.0.127 inclusive.

## File Description

 - MixedDDOS: Simulated normal traffic and DDOS traffic. Only 10 of the files are uploaded to github, though a total of 31 exist.
   One virtual machine ran the normal traffic while another ran the DDOS traffic. DDOS traffic outnumbers normal traffic.
   Normal traffic consisted of a variety of simulated traffic that ran off and on. ITG-Send commands used for normal traffic are as follows:
	  ~~~~Csa -a 10.0.0.1 -t time
	Csi -a 10.0.0.1 -t time
	Quake3 -a 10.0.0.1 -t time
	Telnet -a 10.0.0.1 -t time
	DNS -a 10.0.0.1 -t time
	-T TCP -a 10.0.0.1 -N 1000 200 -n 512 50 -t time
	-T TCP -a 10.0.0.1 -O 1000 -o 512 -t time0
	-T UDP -a 10.0.0.1 -N 1000 200 -n 512 50 -t time
	-T UDP -a 10.0.0.1 -O 1000 -o 512 -t time
	-a 10.0.0.1 -t time~~~~
	Command for attack traffic is: 'hping3 -d 120 -S -w 64 -p 21 --flood -a source_ip destination_ip'.

 - QuickDDOS: Simulated normal traffic and minor DDOS traffic. Goal was to generate a lot of traffic into a small time frame to allow for quick testing. 
   Each normal traffic command only ran for at most 10 seconds. DDOS traffic was limited to 5 attackers at a time. 
   Commands used to simulate traffic are the same as MixedDDOS.