import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('rotation_log.csv')

plt.figure(figsize=(12, 8))
for i in range(3):
    for j in range(3):
        plt.plot(df['frame'], df[f'R_{i}{j}'], label=f'R_{i}{j}')

plt.legend()
plt.xlabel('Frame')
plt.ylabel('Rotation Matrix Element')
plt.title('Rotation Matrix Elements over Time')
plt.show()
