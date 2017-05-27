import time
import threading
import re
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf

from FTPLogReader import FTPLogReader

###While running, every hour this script will read in input from the LogFile "LogFile.txt", parse the input, and then display the input to the user.
###   If you wish to stop LogReader and resume later where you left off, simply save the attribute "position" from the FTPLogReader class and then pass this
###  number back into the LogReader on start up.
### If the log file is large, and is being read all at once, it may take awhile to parse.
__author__ = "Caleb Whitman"
__version__ = "1.0.0"
__email__ = "calebrwhitman@gmail.com"



"""Reads in the logfile and prints the parsed output. Meant to be called every hour."""
def readLogFile(reader):

    print("Here")
    for line in reader.getConnectionTensors():
        print(line)


def main():
    scheduler = BackgroundScheduler()
    reader = FTPLogReader("normal.log", 0)
    job = scheduler.add_job(readLogFile, 'interval', seconds=5,args=[reader])
    scheduler.start()
    input("Enter Any text to quit: ")



if __name__ == '__main__':
   main()