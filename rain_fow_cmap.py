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
    matrix = np.zeros((matrix_size, matrix_size))
    
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

    # Create the rainflow matrix
    rainflow_matrix = create_rainflow_matrix(cycles)

    # Plot the rainflow matrix
    plt.imshow(rainflow_matrix, aspect='auto', cmap='jet', origin='lower', extent=[min(cycles, key=lambda x: x[1])[1], max(cycles, key=lambda x: x[1])[1], 0, max(cycles, key=lambda x: x[0])[0]])
    plt.colorbar(label='Cycle Counts')
    plt.xlabel('Mean')
    plt.ylabel('Range')
    plt.title('Rainflow Matrix')
    plt.grid(True)
    plt.show()
