# Sample Attack Traffic

DOS and DDOS traffic. DDOS traffic was generated using virtual machines. 
Due to limited resources, our DDOS traffic does not completely match DDOS traffic, which is often generated using thousands or millions of machines.
Traffic was captured using tcpdump with the command `tcpdump -i any -w capture_file'.
Due to github's limit on file size, large files are broken down into smaller ones using `tcpdump -r old_file -w new_files -C 100`.

## File Description
 
 - LOICFast: Fast DOS traffic generated using [LOIC](https://sourceforge.net/projects/loic/) for 1 minute 
 (Note: LOIC may contain malware. Don't download on a computer you care about). LOIC settings used are as follows:
 
	port: 21
	
	speed: fastest
	
	threads: 10
	
	method: TCP
	
	timeout: 9001
	
	Wait for Reply: True
	
	Souce IP: 198.168.56.102
	
	Receiving IP: 198.168.56.101
	
 - LOICSlow: Slow DOS traffic generated using [LOIC](https://sourceforge.net/projects/loic/) for 15 minutes 
 (Note: LOIC may contain malware. Don't download on a computer you care about). LOIC settings used are as follows:
 
	port: 21
	
	speed: slowest
	
	threads: 1
	
	method: TCP
	
	timeout: 9001
	
	Wait for Reply: True
	
	Souce IP: 198.168.56.102
	
	Receiving IP: 198.168.56.101
	
 - hping3Fast: Fast DDOS traffic generated using [hping3](https://tools.kali.org/information-gathering/hping3) for 3 minutes using the command `hping3 -d 120 -S -w 64 -p 21 --flood -a source_ip destination_ip`. 
 The attacks were started shortly after tcpdump began capturing traffic.
 Three virtual machines and 12 spoofed IP addresses were used to generated the traffic. Source IP addresses are 192.168.56.103 through 192.168.56.115. Destination IP is 192.168.56.101.
  
 - hpingSlow: Fast DDOS traffic generated using [hping3](https://tools.kali.org/information-gathering/hping3) for 15 minutes using the command `hping3 -d 120 -S -w 64 -p 21 --flood -a source_ip destination_ip`. 
 The attacks were started shortly after tcpdump began capturing traffic.
 Three virtual machines and 12 spoofed IP addresses were used to generated the traffic. Source IP addresses are 192.168.56.103 through 192.168.56.115. Destination IP is 192.168.56.101.