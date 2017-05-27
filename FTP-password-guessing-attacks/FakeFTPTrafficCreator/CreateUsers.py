###From a file, creates new linux users and passwords with administrator permissions

from subprocess import Popen, PIPE, check_call

__author__ = "Caleb Whitman"
__version__ = "1.0.0"
__email__ = "calebrwhitman@gmail.com"


""" Creates new users based off of the given user list
    Args:
        users (string): file containing a list of usernames and passwords. Created by createUsersList.py
        """
def createUsers(users):
    with open(users) as f:
        for line in f.readlines(0):
            namePass = line.split(';')
            user = namePass[0].strip('\n')
            password = namePass[1].strip('\n')
            check_call(['useradd', user])
            proc = Popen(['passwd', password], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            proc.stdin.write('password\n')
            proc.stdin.write('password')
            proc.stdin.flush()
            stdout, stderr = proc.communicate()
            print (stdout)
            print (stderr)

    

if __name__ == '__main__':
    createUsers("users.txt")