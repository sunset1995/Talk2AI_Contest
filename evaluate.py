import sys
import numpy as np

with open(sys.argv[1], 'r') as f:
    f.readline()
    correct_y = np.array([int(line.strip().split(',')[-1]) for line in f])

with open(sys.argv[2], 'r') as f:
    f.readline()
    my_y = np.array([int(line.strip().split(',')[-1]) for line in f])

print('%.4f' % (np.sum(my_y == correct_y) / len(correct_y)))
