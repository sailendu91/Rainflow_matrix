import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_turning_points(data):
    turning_points = []
    for i in range(1, len(data) - 1):
        if (data[i-1] < data[i] > data[i+1]) or (data[i-1] > data[i] < data[i+1]):
            turning_points.append(i)
    return turning_points

def rainflow_counting(data):
    turning_points = find_turning_points(data)
    cycles = []

    while len(turning_points) > 1:
        i = 0
        while i < len(turning_points) - 1:
            if i < len(turning_points) - 2:
                a, b, c = data[turning_points[i]], data[turning_points[i+1]], data[turning_points[i+2]]
                if (a < b > c) or (a > b < c):
                    range_ = abs(b - a) / 2
                    mean = (a + c) / 2
                    cycles.append((range_, mean))
                    turning_points.pop(i+1)
                else:
                    i += 1
            else:
                i += 1
        turning_points.pop(0)

    return cycles

def create_rainflow_matrix(cycles):
    max_range = max(cycles, key=lambda x: x[0])[0]
    max_mean = max(cycles, key=lambda x: abs(x[1]))[1]
    min_mean = min(cycles, key=lambda x: abs(x[1]))[1]
    matrix_size = int(max_range * 100) + 1  # Increase the matrix size based on max range
    mean_range = max_mean - min_mean
    matrix = np.zeros((matrix_size, int(matrix_size * mean_range) + 1))
    
    for cycle in cycles:
        # Convert range and mean to matrix indices (binning)
        range_bin = int(np.floor(cycle[0] * 100))
        mean_bin = int(np.floor((cycle[1] - min_mean) * 100))
        
        # Increment the count at the corresponding matrix cell
        matrix[range_bin, mean_bin] += 1
        
    return matrix

if __name__ == "__main__":
    # Read signal from XLSX file
    file_path = "path_to_your_xlsx_file.xlsx"  # Replace with the actual file path
    df = pd.read_excel(file_path)

    time = df['Time'].values
    signal = df['Amplitude'].values

    # Perform rainflow counting on the signal
    cycles = rainflow_counting(signal)

    if len(cycles) > 0:
        # Create the rainflow matrix for all cycles
        rainflow_matrix = create_rainflow_matrix(cycles)

        # Get scatter plot data from the rainflow matrix
        scatter_x, scatter_y = np.meshgrid(np.linspace(min(cycles, key=lambda x: x[1])[1], max(cycles, key=lambda x: x[1])[1], rainflow_matrix.shape[1]),
                                           np.linspace(0, max(cycles, key=lambda x: x[0])[0], rainflow_matrix.shape[0]))
        scatter_counts = rainflow_matrix.flatten()

        # Plot the rainflow matrix as a scatter plot
        plt.scatter(scatter_x.flatten(), scatter_y.flatten(), c=scatter_counts, cmap='jet')
        plt.colorbar(label='Cycle Counts')
        plt.xlabel('Mean')
        plt.ylabel('Range')
        plt.title('Rainflow Matrix (Scatter Plot)')
        plt.grid(True)
        plt.show()
    else:
        print("No rainflow cycles found.")
