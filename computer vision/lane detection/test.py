data = [(12, 4), (23, 5), (44, 7), (55, 8)]

# Create a set of existing y values
existing_y_values = set(y for _, y in data)

# Find the missing y values
missing_y_values = set(range(min(existing_y_values), max(existing_y_values) + 1)) - existing_y_values
print(missing_y_values)
# Get the highest x value for each missing y value
missing_entries = [(max(x for x, y in data if y == missing_y), missing_y) for missing_y in missing_y_values]

# Append the missing entries to the original data
data += missing_entries

# Sort the data based on y values
data.sort(key=lambda x: x[1])

print(data)