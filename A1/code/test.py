import csv

indir = '../data/';
wordlist_dir = '../../Wordlists/'
BristolGilhoolyLogie_dic = {}

filename = wordlist_dir +'BristolNorms+GilhoolyLogie.csv'
with open(filename,'r') as csvfile:
    header_line = next(csvfile)
    for line in csvfile.readlines():
        col = line.split(',')
        info = []
        info.extend((col[3],col[4],col[5]))
        BristolGilhoolyLogie_dic[col[1]] = info

print(BristolGilhoolyLogie_dic['ability'])
print(BristolGilhoolyLogie_dic['active'])
