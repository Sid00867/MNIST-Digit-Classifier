import numpy as np
import pandas as pd
from MachineLearning import NeuralNetwork
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pickle

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Normalize pixel values to be between 0 and 1
X = X.astype('float32') / 255.0

# Convert labels to one-hot encoding
def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y.astype(int)]

y = to_one_hot(y)

# Create column names for pixels and labels
pixel_columns = [f'pixel_{i}' for i in range(784)]
label_columns = [f'label_{i}' for i in range(10)]

# Create DataFrame
df = pd.DataFrame(X, columns=pixel_columns)
df[label_columns] = y

# Now df has 794 columns: 784 pixel columns + 10 label columns

# =================================================================================

model = NeuralNetwork(
    dataset=df,
    neuron_map=[784, 128, 64, 10],  
    activation_function="relu",      
    output_activation_function="softmax", 
    learning_rate=0.01,           
    cost_function="categorical_cross_entropy",
    absolute_gradient_clipping=1.0,  
    regularization_parameter=0.05
)

model.PrepareData(label_column_names=label_columns,
                  split_ratio_training=0.8, 
                #   Architecture_test=True,
                #   A_test_samples=1000
                    )

(costs,test_costs,pure_costs,
    [neuron_map_o, 
    hidden_activation_type_o, 
    output_activation_type_o, 
        [hidden_layer_weights_o,
        hidden_layer_biases_o,
        output_neurons_weights_o,
        output_neurons_biases_o]
    ]) = model.train(
            epochs=50,           
            batch_size=64,                                  
            verbose=True,        
            print_every=5
    )        

model.test()

# Save model information
model_info = {
    'neuron_map': neuron_map_o,
    'hidden_activation': hidden_activation_type_o,
    'output_activation': output_activation_type_o,
    'hidden_weights': hidden_layer_weights_o,
    'hidden_biases': hidden_layer_biases_o,
    'output_weights': output_neurons_weights_o,
    'output_biases': output_neurons_biases_o
}

# Save to file (will override if exists)
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

# Plot the training and test costs
plt.figure(figsize=(10, 6))
plt.plot(pure_costs, label='Training Cost')
plt.plot(test_costs, label='Test Cost')
plt.plot(costs, label='Regularized Training Cost')
plt.title('Training & Test Cost Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.show()


