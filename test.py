from tensorflow.keras.utils import to_categorical
import numpy as np

y = np.array([0, 1, 2])
y_one_hot = to_categorical(y, num_classes=3)
print(y_one_hot)
