import numpy as np

# Create a 100x100 array filled with zeros
map = np.zeros((100, 100))

# Set the values of the map array to 1 for the rows 50 to 55 and columns 0 to 100
map[50:55, 0:100] = 1

# Print the full sample array
print(map)