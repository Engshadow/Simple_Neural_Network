import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from Neural_Network import SimpleNeuralNetwork

def load_images_from_folder(folder_path, label, img_size=(64, 64)):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        try:
            img = Image.open(path).convert("RGB")     
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0        
            X.append(img_array.flatten())          
            y.append(label)
        except:
            pass
    return X, y

X_cancer_train, y_cancer_train = load_images_from_folder("Skin_Data/Cancer/Training", 1)
X_non_train,    y_non_train    = load_images_from_folder("Skin_Data/Non_Cancer/Training", 0)

X_cancer_test, y_cancer_test = load_images_from_folder("Skin_Data/Cancer/Testing", 1)
X_non_test,    y_non_test    = load_images_from_folder("Skin_Data/Non_Cancer/Testing", 0)

X = np.array(X_cancer_train + X_non_train)
y = np.array(y_cancer_train + y_non_train).reshape(-1, 1)

X_test = np.array(X_cancer_test + X_non_test)
y_test = np.array(y_cancer_test + y_non_test).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

num_hidden_layers = int(input("Enter number of hidden layers: "))

hidden_layer_sizes = []
activations = []

input_size = X_train.shape[1]   
output_size = 1              

for i in range(num_hidden_layers):
    neurons = int(input(f"Enter neurons in hidden layer {i+1}: "))
    hidden_layer_sizes.append(neurons)
    act = input(f"Activation for layer {i+1} (relu/sigmoid): ").lower()
    activations.append(act)

hidden_layer_sizes.append(output_size)
activations.append("sigmoid")

layer_sizes = [input_size] + hidden_layer_sizes

nn = SimpleNeuralNetwork(layer_sizes=layer_sizes, activations=activations)

learning_rate = float(input("Enter learning rate: "))
num_epochs = int(input("Enter number of epochs: "))

for epoch in range(num_epochs):
    nn.forward(X_train)
    nn.backward(X_train, y_train, learning_rate)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        y_pred_train = nn.forward(X_train)
        train_loss = -np.mean(y_train*np.log(y_pred_train + 1e-8) + (1 - y_train)*np.log(1 - y_pred_train + 1e-8))

        y_pred_val = nn.forward(X_val)
        val_loss = -np.mean(y_val*np.log(y_pred_val + 1e-8) + (1 - y_val)*np.log(1 - y_pred_val + 1e-8))

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("\n--- Test Set Predictions ---")
nn.predict(X_test)
