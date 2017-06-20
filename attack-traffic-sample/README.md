# Sample Attack Traffic

DOS and DDOS traffic. DDOS traffic has not been generated yet. 
Traffic was captured using tcpdump with the command `tcpdump -n -nn -i any -w capture_file'.

## File Description
 
 - LOICFast: Fast DOS traffic generated using [LOIC](https://sourceforge.net/projects/loic/)for 1 minute 
 (Note: LOIC may contain malware. Don't download on a computer you care about). LOIC settings used are as follows:
 
	port: 21
	
	speed: fastest
	
	threads: 10
	
	method: TCP
	
	timeout: 9001
	
	Wait for Reply: True
	
	Souce IP: 198.168.56.102
	
	Receiving IP: 198.168.56.101
	
 - LOICSlow: Slow DOS traffic generated using [LOIC](https://sourceforge.net/projects/loic/)for 15 minutes 
 (Note: LOIC may contain malware. Don't download on a computer you care about). LOIC settings used are as follows:
 
	port: 21
	
	speed: slowest
	
	threads: 1
	
	method: TCP
	
	timeout: 9001
	
	Wait for Reply: True
	
	Souce IP: 198.168.56.102
	
	Receiving IP: 198.168.56.101