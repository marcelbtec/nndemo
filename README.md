# Neural Network Visualization Demo

An interactive educational tool for visualizing and understanding how neural networks learn to classify data. This React-based application demonstrates the training process of a simple neural network through various classification tasks.

## 🎯 Learning Objectives

This demo helps you understand:
- How neural networks learn to classify data
- The impact of different activation functions (Sigmoid vs ReLU)
- How network architecture affects learning
- The concept of decision boundaries
- Different types of classification problems (linear, circular, XOR, spiral)

## 🚀 Features

- **Interactive Training**: Watch the network learn in real-time
- **Multiple Datasets**: Experiment with different types of classification problems:
  - Linear separation
  - Circular separation
  - XOR problem
  - Spiral separation
- **Customizable Parameters**:
  - Number of hidden units
  - Learning rate
  - Activation function (Sigmoid/ReLU)
- **Visual Feedback**:
  - Real-time decision boundary visualization
  - Training progress indicators
  - Network architecture visualization

## 🛠️ Technical Implementation

The project implements a simple neural network with:
- Input layer (2 neurons)
- Hidden layer (configurable number of neurons)
- Output layer (1 neuron)
- Backpropagation with momentum
- Support for both Sigmoid and ReLU activation functions

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/marcelbtec/nndemo.git
cd nndemo
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

## 🎮 How to Use

1. Select a dataset type from the dropdown menu
2. Adjust the network parameters:
   - Number of hidden units
   - Learning rate
   - Activation function
3. Click "Train" to start the learning process
4. Observe how the decision boundary changes during training
5. Experiment with different parameters to see their effects

## 📚 Educational Value

This demo is particularly useful for:
- Understanding the basics of neural networks
- Visualizing the learning process
- Experimenting with different network architectures
- Learning about activation functions
- Understanding the challenges of different types of classification problems

## 🧠 Key Concepts Demonstrated

1. **Forward Propagation**: How inputs flow through the network
2. **Backpropagation**: How the network learns from errors
3. **Decision Boundaries**: How the network separates different classes
4. **Activation Functions**: The role of different activation functions
5. **Learning Rate**: How it affects the training process
6. **Network Architecture**: The impact of hidden layer size

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new dataset types
- Implement additional visualization features
- Improve the educational content
- Fix bugs or optimize performance

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with React and Create React App
- Uses Lucide React for icons
- Inspired by educational neural network visualizations
