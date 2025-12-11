import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm  
from Neural_Network import SimpleNeuralNetwork

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def load_images_from_folder(folder_path, label, img_size=(64, 64)):
    X_data = []
    y_data = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return [], []

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        try:
            img = Image.open(path)
            
            img = img.convert("L")
            
            img = img.resize(img_size)
            
            # Normalize pixel values to [0, 1]
            img_array = np.array(img) / 255.0
            
            X_data.append(img_array.flatten())
            y_data.append(label)
        except Exception as e:
            pass
            
    return X_data, y_data

base_dir = "PandasBears"

X_pandas_train, y_pandas_train = load_images_from_folder(f"{base_dir}/train/pandas", 1)
X_bears_train,  y_bears_train  = load_images_from_folder(f"{base_dir}/train/bears", 0)

X_pandas_test, y_pandas_test = load_images_from_folder(f"{base_dir}/test/pandas", 1)
X_bears_test,  y_bears_test  = load_images_from_folder(f"{base_dir}/test/bears", 0)

X = np.array(X_pandas_train + X_bears_train)
y = np.array(y_pandas_train + y_bears_train).reshape(-1, 1)

X_test = np.array(X_pandas_test + X_bears_test)
y_test = np.array(y_pandas_test + y_bears_test).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

if X_train.shape[0] == 0:
    print("Error: No images loaded.")
else:
    input_size = X_train.shape[1]
    output_size = 1
    print("Program started ")
    try:
        num_hidden_layers = int(input("Enter number of hidden layers: "))
    except ValueError:
        print("Invalid input Using default value of 1.")
        num_hidden_layers = 1
    hidden_layer_sizes = []
    activations = []
    for i in range(num_hidden_layers):
        try:
            neurons = int(input(f"Enter neurons in hidden layer {i+1} "))
            act = input(f"Activation for layer {i+1} (relu/sigmoid) ").lower()
        except:
            neurons = 80
            act = 'relu'
        hidden_layer_sizes.append(neurons)
        activations.append(act)
    #output layer
    hidden_layer_sizes.append(output_size)
    activations.append("sigmoid")
    layer_sizes = [input_size] + hidden_layer_sizes
    nn = SimpleNeuralNetwork(layer_sizes=layer_sizes, activations=activations)
    try:
        learning_rate = float(input("Learning rate : "))
        num_epochs = int(input("Epochs : "))
    except:
        learning_rate = 0.01
        num_epochs = 100

    print("\n Starting Training ...")
    epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        nn.forward(X_train)
        nn.backward(X_train, y_train, learning_rate)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            y_pred_train = nn.forward(X_train)
            loss = mse_loss(y_train, y_pred_train)
            
            epoch_bar.set_postfix({"MSE Loss": f"{loss:.4f}"})


    print("\n Evaluation on Test Set ")
    predictions = nn.predict(X_test)
    
    correct_predictions = np.sum(predictions.flatten() == y_test.flatten())
    total_samples = len(y_test)
    accuracy = (correct_predictions / total_samples) * 100
    
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")


            



