import torch


a = {'a':1, 'b':2}

file = open('test.txt', 'w', encoding = 'utf8')

for key, value in a.items():
    file.writelines(str(key) + ' ' + str(value) + '\n')

file.close()


