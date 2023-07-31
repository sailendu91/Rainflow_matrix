import numpy as np
import matplotlib.pyplot as plt

def create_random_signal(duration, sample_rate, amplitude):
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    signal = np.random.randn(num_samples) * amplitude
    return time, signal

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
    duration_sec = 600
    sample_rate_hz = 100  # Samples per second
    amplitude = 1.0

    time, signal = create_random_signal(duration_sec, sample_rate_hz, amplitude)

    # Perform rainflow counting on the signal
    cycles = rainflow_counting(signal)

    if len(cycles) > 0:
        # Extract the first cycle from the rainflow cycles
        first_cycle = [cycles[0]]

        # Create the rainflow matrix for the first cycle
        rainflow_matrix = create_rainflow_matrix(first_cycle)

        # Get scatter plot data from the rainflow matrix
        scatter_x, scatter_y = np.where(rainflow_matrix > 0)
        scatter_counts = rainflow_matrix[scatter_x, scatter_y]

        # Plot the scatter plot of the first rainflow cycle
        plt.scatter(scatter_y, scatter_x, c=scatter_counts, cmap='jet')
        plt.colorbar(label='Cycle Counts')
        plt.xlabel('Mean Bins')
        plt.ylabel('Range Bins')
        plt.title('First Rainflow Cycle Scatter Plot')
        plt.grid(True)
        plt.show()
    else:
        print("No rainflow cycles found.")
