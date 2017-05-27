# Give a username list and a password list, this script creates a list of n usernames and passwords.
# Username list gotten from https://github.com/maryrosecook/commonusernames
# Password list is from Kali Linux.

import random

__author__ = "Caleb Whitman"
__version__ = "1.0.0"
__email__ = "calebrwhitman@gmail.com"


"""Gets n random lines from the passed in file.
   Note: This method loads the entire file into memory.
    Args:
        afile (string): the files to get the random lines
        n (int): the number of randomg lines to be gotten.
    returns: a random line in the form of a string. """
def random_lines(afile,n):
    lines = []
    with open(afile, encoding="latin-1") as f:
        all_lines = f.readlines()
        for i in range(0,n):
            lines.append(random.choice(all_lines).strip(' \t\n\r'))
    return lines

"""Gets n random unique lines from the passed in file.
    Note: This method loads the entire file into memory.
    Args:
        afile (string): the file name to get the files.
        n (int): the number of lines to be loaded.
    returns: a list of strings for each line."""
def random_lines_unique(afile,n):
    with open(afile, encoding="latin-1") as f:
        lines = random.sample(f.readlines(), n)
        for i in range(0,len(lines)):
            lines[i] = lines[i].strip(' \t\n\r')
    return lines


""" Instanitates a new FTPLogReader.
    Args:
        users (string): file containing a list of usernames
        passwords (string): file containing a list of passwords
        user_num (int): number of users to be written to the outfile
        outfile (string): The file where the username and password list will be written to.
        """
def createUserList(users_file,passwords_file,user_num,result_file):

    users = random_lines_unique(users_file,user_num)
    passwords = random_lines(passwords_file,user_num)

    with open(result_file,'w') as f:
        for i in range(0,len(users)):
            f.write(users[i]+";"+passwords[i]+"\n")


if __name__ == '__main__':
    createUserList("usernames.txt","rockyou.txt",100,"users.txt")




