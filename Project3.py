#!/usr/bin/env python3
'''
Program to train different neurons to predict the power consumption for a particular hour in the day
'''
import numpy as np
import pandas as pd

# class to represent the neurons where degree is the degree of the polynomail ex. y = x^2 is a 2nd degree polynomial
class Neuron():
    def __init__(self, degree):
        self.degree = degree
        self.initialize_weights()

    # for every term in polynomial assign a random weight between 1 and 0
    def initialize_weights(self):
        w = []
        for i in range(self.degree + 1):
            w.append(np.random.random())
        self.weights = w

    def net(self,x):
        sum = 0
        for i in range(self.degree + 1):
            sum = sum + self.weights[i]*x**i
        return sum

    def output(self,x, k):
        return k*self.net(x)

    def perc_learn(self,inp,out):
        alpha = 0.5
        iterations = 10000
        count = 0
        while iterations > count:
            for i in range(len(inp)):
                delta_w = []
                for w in range(len(self.weights)):
                    delta_w.append(alpha*(inp[i]**w)*(out[i]-self.output(inp[i],1)))
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] + delta_w[j]
            count = count + 1
def calc_TE(neuron, inp, out):
    te = 0
    for i in range(len(inp)):
        pred = neuron.output(inp[i], 1)
        te = (out[i] - pred)**2


def read_csv(filename):
    df = pd.read_csv(filename)
    x = df['time']
    y = df['power']
    return x,y

def main():
    train_files = ["train_norm1.csv", "train_norm2.csv", "train_norm3.csv"]
    for i in range(len(train_files)):
        inp,out = read_csv(train_files[i])
        inp = np.array(inp)
        out = np.array(out)

if __name__=='__main__':
    main()
