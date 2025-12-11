from sklearn.datasets import fetch_openml # downloads datasets from the OpenML online repository.
import numpy as np # used for numarical operations, storing images as arrays, reshaping, type conversions, etc.
from sklearn.model_selection import train_test_split # splits data into training, validation, and testing sets randomly
from sklearn.preprocessing import OneHotEncoder # converts integer labels (0–9) into one-hot encoded vectors

# Load MNIST from OpenML
# parameter 1: the name of the data we want to fetch
# parameter 2: the vesion of the data we want to fetch
# parameter 3: false to return data as numpy arrays instead of pands dataframes
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data # image collection
y = mnist.target.astype(int) # int lables from 0 to 9


# normalize pixels to range [0, 1]
# NNs train much better when input between 0 and 1
X = X.astype("float32") / 255.0

# reshap to ensure valid shape and works even if data changes
num_samples = X.shape[0]
X = X.reshape(num_samples, 28*28)


# one-Hot Encode the labels
# OneHotEncoder is a preprocessing technique used to convert categorical variables into a numerical format suitable for ML algorithms
# sparse_output=False returns a dense NumPy array
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))


# 70% train, 30% temp
# random_state=42 guarantees same random split every time
# randomize before splitting
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=True
)

# Split temp into 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

# Print shapes to verify everything
print("Train data shape: ", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test data shape: ", X_test.shape, y_test.shape)