__author__ = 'admin'

import numpy as np
import csv as csv


def read_from_csv(filename):
    csv_file = csv.reader(open(filename, 'rb'))
    header = csv_file.next()

    data = []
    for row in csv_file:
        elements = map(int, row)
        data.append(elements)

    data = np.array(data)

    return data

def write_to_csv(data, filename, header='header'):
    csv_file = csv.writer(open(filename, 'wb'))
    if header != None:
        csv_file.writerow('header')

    for i in range(data.shape[0]):
        csv_file.writerow(data[i])

def write_predicted_result(data, filename):
    fp = open(filename, "wb")
    fp.write("ImageId,Label\n")
    for index, item in enumerate(data):
        fp.write("%d,%s\n" % (index+1, item))
    fp.close()
