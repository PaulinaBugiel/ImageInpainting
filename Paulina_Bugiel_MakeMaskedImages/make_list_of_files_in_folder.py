# Author: Paulina Bugiel, 2024

import os
import sys


""" Helper script for creating a list of files in a directory. If desired, 
 only every n-th file path can be written by specifying the increment argument """

def main(directory, increment: int = 1):
    directory = os.path.normpath(directory.strip('"\''))
    if increment > 1:
        list_path = os.path.normpath(directory) + '_list_every_' + str(increment) + '.txt'
    elif increment == 1:
        list_path = os.path.normpath(directory) + '_list.txt'
    else:
        print('Invalid increment. Should be greater or equal to 1')
        return

    cnt_all = 0
    cnt_written = 0
    with open(list_path, 'w') as list_file:
        for file in os.listdir(directory):
            if cnt_all % increment == 0:
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    list_file.write(file_path + '\n')
                    cnt_written += 1
                    print(f'Written {cnt_written:8}/{cnt_all:8}', end='\r')
            cnt_all += 1
    print('\nDone! Written', cnt_written, ' files to list (out of', cnt_all, ')')


if __name__ == '__main__':
    # percentage is the default option to run

    main(sys.argv[1], int(sys.argv[2]))
