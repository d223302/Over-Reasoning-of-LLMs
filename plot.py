#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Names of models
models = ['ChatGPT', 'Llama']

# Values for LIMA wins, Ties, and LIMA Loses
wins = [53, 44, 33, 24, 18]
ties = [21, 21, 25, 22, 25]
loses = [26, 35, 42, 54, 57]

barWidth = 0.85

# Set the position of the bars on x axis
r = np.arange(len(models))

# Create blue bars
plt.bar(r, wins, color='blue', edgecolor='white', width=barWidth, label='LIMA wins')
# Create cyan bars (middle), on top of the first ones
plt.bar(r, ties, bottom=wins, color='cyan', edgecolor='white', width=barWidth, label='Tie')
# Create navy bars (top)
plt.bar(r, loses, bottom=[i+j for i,j in zip(wins, ties)], color='navy', edgecolor='white', width=barWidth, label='LIMA Loses')

# Custom x axis
plt.xticks(r, models, rotation=45)
plt.ylabel("Percentage")
plt.title("Comparison Chart")
plt.legend(loc="upper left", bbox_to_anchor=(1,1))

# Show the graph
plt.tight_layout()
plt.show()

