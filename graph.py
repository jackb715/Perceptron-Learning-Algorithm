#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import sys
from Project3 import Neuron

df = pd.read_csv("train_norm1.csv")
df2 = pd.read_csv("train_norm2.csv")
df3 = pd.read_csv("train_norm3.csv")
x = df['time']
y = df['power']
y2 = []
n = Neuron(3)
n.perc_learn(x,y)


for inp in x:
    y2.append(n.output(inp,1))

plt.plot(x,y2)
plt.scatter(x,y)
plt.show()
