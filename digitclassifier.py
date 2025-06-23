import pygame
import numpy as np
import sys
import pickle
import pandas as pd

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 28
CELL_SIZE = 20
GRID_WIDTH = GRID_SIZE * CELL_SIZE
GRID_HEIGHT = GRID_SIZE * CELL_SIZE
PANEL_WIDTH = 200
WINDOW_WIDTH = GRID_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = GRID_HEIGHT
BACKGROUND_COLOR = (240, 240, 240)
GRID_COLOR = (200, 200, 200)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CHECKBOX_SIZE = 15

class MNISTInterface:
    def __init__(self):

        with open('./mnist_model.pkl', 'rb') as f:
            self.model_info = pickle.load(f)

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("MNIST Digit Classifier")
        self.clock = pygame.time.Clock()
        
        # Initialize the grid with zeros (white background)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        
        # Font for displaying prediction
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Prediction variables
        self.predicted_char = "?"
        self.confidence = 0.0
        self.probabilities = None
        
        # Drawing variables
        self.drawing = False
        self.brush_size = 1
        self.greyscale_mode = True  # Toggle between greyscale and pure black
        
        # Checkbox position
        self.checkbox_rect = pygame.Rect(GRID_WIDTH + 10, 500, CHECKBOX_SIZE, CHECKBOX_SIZE)
        
    def your_neural_network_function(self, X):
        # X_batch is a numpy array of shape (m, n), where m is samples, n is features

        # print(X.shape)
        # Convert X (should be shape (n_samples, 784) or (784,) or (1,784)) to a DataFrame with 784 columns
        # If X is 1D, reshape to (1, 784)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        pixel_columns = [f'pixel_{i}' for i in range(784)]
        X_df = pd.DataFrame(X, columns=pixel_columns)
        X_batch = X_df.values  # X_batch shape: (1, 784) if single sample

        self.z_values = []
        self.a_values = []

        current_a = X_batch.T # Shape (features, samples)

        # Hidden layers
        hidden_weights = self.model_info['hidden_weights']
        hidden_biases = self.model_info['hidden_biases']
        neuron_map = self.model_info['neuron_map']
        hidden_activation_type = self.model_info['hidden_activation']
        output_activation_type = self.model_info['output_activation']
        output_weights = self.model_info['output_weights']
        output_biases = self.model_info['output_biases']

        num_hidden_layers = len(neuron_map) - 2

        # Hidden layers
        for i in range(num_hidden_layers):
            z = np.dot(hidden_weights[i], current_a) + hidden_biases[i]
            current_a = self._apply_activation(z, hidden_activation_type)
            self.z_values.append(z)
            self.a_values.append(current_a)

        # Output layer
        # If no hidden layers, current_a is still X_batch.T
        # If hidden layers, current_a is the activation of the last hidden layer
        z_output = np.dot(output_weights, current_a) + output_biases
        a_output = self._apply_activation(z_output, output_activation_type)
        self.z_values.append(z_output)
        self.a_values.append(a_output)

        # a_output shape: (num_output_neurons, num_samples)
        # print(a_output)
        return a_output
    
    def _apply_activation(self, z, activation_type):
        if activation_type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif activation_type == "tanh":
            return np.tanh(z)
        elif activation_type == "relu":
            return np.maximum(0, z)
        elif activation_type == "linear":
            return z
        elif activation_type == "softmax":
            # Subtract max for numerical stability
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    def get_character_from_index(self, index):
        """Convert probability array index to character."""
        return str(index)  # Digits 0-9 only
    
    def update_prediction(self):
        """Update the prediction based on current grid state."""
        # Flatten and normalize the grid
        pixel_values = self.grid.flatten().astype(np.float32) / 255.0
        
        # Get probabilities from your neural network
        self.probabilities = self.your_neural_network_function(pixel_values)

        # Get the predicted character
        predicted_index = np.argmax(self.probabilities)
        # Ensure confidence is a scalar float
        prob_val = self.probabilities[predicted_index]
        if isinstance(prob_val, np.ndarray):
            self.confidence = float(np.squeeze(prob_val))
        else:
            self.confidence = float(prob_val)
        self.predicted_char = self.get_character_from_index(predicted_index)
    
    def draw_on_grid(self, pos):
        """Draw on the grid at the given position."""
        x, y = pos
        if x < GRID_WIDTH and y < GRID_HEIGHT:
            grid_x = x // CELL_SIZE
            grid_y = y // CELL_SIZE
            
            # Draw with brush (circular area)
            for dy in range(-self.brush_size, self.brush_size + 1):
                for dx in range(-self.brush_size, self.brush_size + 1):
                    if dx*dx + dy*dy <= self.brush_size*self.brush_size:
                        new_x = grid_x + dx
                        new_y = grid_y + dy
                        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                            if self.greyscale_mode:
                                # Gradually darken the cell (additive drawing)
                                current_value = self.grid[new_y, new_x]
                                self.grid[new_y, new_x] = min(255, current_value + 80)
                            else:
                                # Pure black drawing
                                self.grid[new_y, new_x] = 255
    
    def clear_grid(self):
        """Clear the drawing grid."""
        self.grid.fill(0)
        self.predicted_char = "?"
        self.confidence = 0.0
        self.probabilities = None
    
    def draw_grid(self):
        """Draw the 28x28 grid."""
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Color based on pixel value (0=white, 255=black)
                gray_value = 255 - self.grid[y, x]  # Invert for display
                color = (gray_value, gray_value, gray_value)
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)
    
    def draw_checkbox(self, surface, rect, checked):
        """Draw a checkbox."""
        # Draw checkbox border
        pygame.draw.rect(surface, BLACK, rect, 2)
        
        # Fill if checked
        if checked:
            inner_rect = pygame.Rect(rect.x + 3, rect.y + 3, rect.width - 6, rect.height - 6)
            pygame.draw.rect(surface, BLACK, inner_rect)
    
    def draw_panel(self):
        """Draw the right panel with prediction and controls."""
        panel_x = GRID_WIDTH
        
        # Background
        panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, panel_rect)
        
        # Title
        title_text = self.font_medium.render("Prediction", True, BLACK)
        self.screen.blit(title_text, (panel_x + 10, 20))
        
        # Predicted character (large)
        char_text = self.font_large.render(f"{self.predicted_char}", True, BLACK)
        char_rect = char_text.get_rect(center=(panel_x + PANEL_WIDTH//2, 100))
        self.screen.blit(char_text, char_rect)
        
        # Confidence
        if self.probabilities is not None:
            conf_text = self.font_small.render(f"Confidence: {self.confidence:.2%}", True, BLACK)
            self.screen.blit(conf_text, (panel_x + 10, 140))
            
            # Top 3 predictions
            flat_probs = np.squeeze(self.probabilities)
            top_indices = np.argsort(flat_probs)[-3:][::-1]
            y_offset = 180
            self.screen.blit(self.font_small.render("Top 3:", True, BLACK), (panel_x + 10, y_offset))
            
            for i, idx in enumerate(top_indices):
                char = self.get_character_from_index(idx)
                prob = flat_probs[idx]
                prob = float(prob)  # Ensure scalar
                text = f"{i+1}. {char}: {prob:.1%}"
                pred_text = self.font_small.render(text, True, BLACK)
                self.screen.blit(pred_text, (panel_x + 10, y_offset + 30 + i*25))
        
        # Instructions
        instructions = [
            "Controls:",
            "Left Click: Draw",
            "Right Click: Erase",
            "C: Clear grid",
            "1-3: Brush size",
            "",
            "Draw a digit (0-9)"
        ]
        
        y_start = 320
        for i, instruction in enumerate(instructions):
            if instruction == "":
                continue
            text = self.font_small.render(instruction, True, BLACK)
            self.screen.blit(text, (panel_x + 10, y_start + i*20))
        
        # Greyscale mode checkbox
        self.draw_checkbox(self.screen, self.checkbox_rect, self.greyscale_mode)
        checkbox_label = self.font_small.render("Greyscale Mode", True, BLACK)
        self.screen.blit(checkbox_label, (self.checkbox_rect.x + CHECKBOX_SIZE + 5, self.checkbox_rect.y - 2))
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if clicking on checkbox
                    if self.checkbox_rect.collidepoint(event.pos):
                        self.greyscale_mode = not self.greyscale_mode
                    else:
                        self.drawing = True
                        self.draw_on_grid(event.pos)
                        self.update_prediction()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    self.drawing = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing:
                    self.draw_on_grid(event.pos)
                    self.update_prediction()
                elif pygame.mouse.get_pressed()[2]:  # Right click drag (erase)
                    x, y = event.pos
                    if x < GRID_WIDTH and y < GRID_HEIGHT:
                        grid_x = x // CELL_SIZE
                        grid_y = y // CELL_SIZE
                        for dy in range(-self.brush_size, self.brush_size + 1):
                            for dx in range(-self.brush_size, self.brush_size + 1):
                                if dx*dx + dy*dy <= self.brush_size*self.brush_size:
                                    new_x = grid_x + dx
                                    new_y = grid_y + dy
                                    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                                        self.grid[new_y, new_x] = max(0, self.grid[new_y, new_x] - 80)
                        self.update_prediction()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Clear
                    self.clear_grid()
                elif event.key == pygame.K_1:  # Brush size 1
                    self.brush_size = 1
                elif event.key == pygame.K_2:  # Brush size 2
                    self.brush_size = 2
                elif event.key == pygame.K_3:  # Brush size 3
                    self.brush_size = 3
                elif event.key == pygame.K_g:  # Toggle greyscale mode
                    self.greyscale_mode = not self.greyscale_mode
        
        return True
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()
            
            # Clear screen
            self.screen.fill(WHITE)
            
            # Draw components
            self.draw_grid()
            self.draw_panel()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    interface = MNISTInterface()
    interface.run()