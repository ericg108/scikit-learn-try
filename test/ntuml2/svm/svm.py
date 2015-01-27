__author__ = 'nenggong'


def phi1(tup):
    return tup[1] * tup[1] - 2 * tup[0] + 3


def phi2(tup):
    return tup[0] * tup[0] - 2 * tup[1] - 3


def transform(tups):
    for tup, label in tups:
        print (phi1(tup), phi2(tup)), label


tups = [((1, 0), -1), ((0, 1 ), -1), ((0, -1 ), -1), ((-1, 0), 1), ((0, 2), 1), ((0, -2), 1), ((-2, 0), 1) ]
transform(tups)