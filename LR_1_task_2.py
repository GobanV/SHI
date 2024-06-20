import numpy as np
from sklearn.preprocessing import Binarizer, scale, normalize

input_data = np.array([4.3, -9.9, -3.5, -2.9, 4.1, 3.3, -2.2, 8.8, -6.1, 3.9, 1.4, 2.2])
threshold = 2.2

binarizer = Binarizer(threshold=threshold)
binary_data = binarizer.fit_transform(input_data.reshape(1, -1))
print("Binary Data:\n", binary_data)

mean_excluded_data = input_data - np.mean(input_data)
print("\nMean Excluded Data:\n", mean_excluded_data)

scaled_data = scale(input_data)
print("\nScaled Data:\n", scaled_data)

normalized_data = normalize(input_data.reshape(1, -1))
print("\nNormalized Data:\n", normalized_data)
