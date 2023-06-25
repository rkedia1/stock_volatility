import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('hyper.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = df['n_estimators']
y = df['learning_rate']
z = df['score']
ax.scatter(x, y, z)

ax.set_xlabel('n_estimators')
ax.set_ylabel('learning_rate')
ax.set_zlabel('score')

plt.show()