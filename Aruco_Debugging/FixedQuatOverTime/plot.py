import csv
import matplotlib.pyplot as plt

# Load quaternion data
frames = []
qx, qy, qz, qw = [], [], [], []

with open('solvepnp_param_change_log.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        frames.append(int(row['frame']))
        qx.append(float(row['qx']))
        qy.append(float(row['qy']))
        qz.append(float(row['qz']))
        qw.append(float(row['qw']))

# Plot each quaternion component
plt.figure(figsize=(12, 8))

plt.plot(frames, qx, label='qx')
plt.plot(frames, qy, label='qy')
plt.plot(frames, qz, label='qz')
plt.plot(frames, qw, label='qw')

plt.xlabel('Frame')
plt.ylabel('Quaternion Value')
plt.title('Quaternion Components Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
