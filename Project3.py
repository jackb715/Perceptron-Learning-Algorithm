#!/usr/bin/env python3
'''
Program to train different neurons to predict the power consumption for a particular hour in the day
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# class to represent the neurons where degree is the degree of the polynomail ex. y = x^2 is a 2nd degree polynomial
class Neuron():
    def __init__(self, degree):
        self.degree = degree
        self.initialize_weights()

    # for every term in polynomial assign a random weight between 1 and 0
    def initialize_weights(self):
        w = []
        for i in range(self.degree + 1):
            #w.append(np.random.randint(-100,100))
            w.append(1)
        self.weights = w

    def net(self,x):
        sum = 0
        for i in range(self.degree + 1):
            sum = sum + self.weights[i]*x**i
        return sum

    def output(self,x, k):
        return k*self.net(x)

    def perc_learn(self,inp,out):
        alpha = 0.4
        iterations = 50000
        count = 0
        te = 1000
        while te > 0.05 and count < iterations:
            for i in range(len(inp)):
                delta_w = []
                for w in range(len(self.weights)):
                    delta_w.append(alpha*(inp[i]**w)*(out[i]-self.output(inp[i],1)))
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] + delta_w[j]
            count = count + 1
            te = self.calc_TE(inp,out)
        return te

    def calc_TE(self, inp, out):
        te = 0
        for i in range(len(inp)):
            pred = self.output(inp[i], 1)
            te = te + (out[i] - pred)**2
        return te


def read_csv(filename):
    df = pd.read_csv(filename)
    x = df['time']
    y = df['power']
    return x,y

def create_plt(neuron, inp, out, title):
    # generate neuron's preictions
    y2 = []
    for x in inp:
        y2.append(neuron.output(x,1))
    fig = plt.figure()
    for o in out:
        plt.scatter(inp,o) # plot data points from set(s)
    plt.plot(inp,y2) # plot perceptron output as function
    plt.title(title)
    fig.savefig(title + ".png")

def main():

    train_files = ["train_norm1.csv", "train_norm2.csv", "train_norm3.csv"]
    test_file = "test_norm.csv"
    neurons = [Neuron(1),Neuron(2),Neuron(3)]
    neuron_names = ["A", "B", "C"]

    test_inp,test_out = read_csv(test_file)
    test_inp = np.array(test_inp)
    test_out = np.array(test_out)

    '''
    For each neuron train on all of the training sets and report the total error for each. Once trained on every dataset test neuron architecture on testing set and report total error
    '''
    count = 0
    for n in neurons:
        total_out = [] # power outputs for all of the testing data
        for i in range(len(train_files)):
            inp,out = read_csv(train_files[i])
            inp = np.array(inp)
            out = np.array(out)
            total_out.append(out)

            te = n.perc_learn(inp,out)
            print("Total error for neuron " + neuron_names[count] + " was " + str(round(te,4)) + " for " + train_files[i])
            print("\n")
        test_te = n.calc_TE(test_inp, test_out)
        print("Total error for neuron " + neuron_names[count] + " was " + str(round(test_te,4)) + " for the testing data")

        title_base = "Neuron " + neuron_names[count]
        create_plt(n, np.array(read_csv(train_files[0])[0]), total_out, (title_base + " with train data"))
        create_plt(n, np.array(read_csv(train_files[0])[0]), [(test_out)], (title_base + " with test data"))
        count+=1


if __name__=='__main__':
    main()
