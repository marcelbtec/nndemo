import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const Documentation = () => {
  const content = `
# Neural Network Visualization Demo

This interactive demo helps you understand how neural networks learn to classify data. Let's explore how it works!

## The Classification Problem

The demo shows a simple binary classification task where we want to separate two classes of points (blue and red) in a 2D space. The neural network learns to draw a decision boundary that separates these classes.

## Network Architecture

The neural network has:
- 2 input neurons (x₁, x₂) representing the coordinates
- A hidden layer with configurable number of neurons
- 1 output neuron (y) that predicts the class

## How to Use the Demo

1. **Choose a Dataset**
   - Linear: Points separated by a straight line
   - Circular: Points separated by a circle
   - XOR: Points arranged in an XOR pattern
   - Spiral: Points arranged in a spiral pattern

2. **Adjust Network Parameters**
   - Hidden Units: Number of neurons in the hidden layer
   - Learning Rate: How quickly the network learns
   - Activation Function: Choose between ReLU and Sigmoid

3. **Training**
   - Click "Train" to start learning
   - Watch the decision boundary evolve
   - Monitor the loss and epoch count
   - Click "Stop" to pause training
   - Click "Reset" to start over

## What to Observe

- The decision boundary (colored region) shows how the network separates the classes
- The network visualization shows the weights between neurons
- Blue lines represent positive weights
- Red lines represent negative weights
- Thicker lines indicate stronger weights

## Learning Process

The network learns by:
1. Forward pass: Making predictions
2. Calculating loss: Measuring prediction error
3. Backpropagation: Adjusting weights to reduce error
4. Repeating until the decision boundary fits the data

## Tips for Different Datasets

- Linear: Works well with few hidden units
- Circular: Needs more hidden units to learn the curve
- XOR: Requires multiple hidden units to learn the pattern
- Spiral: Most complex, needs many hidden units and training time

## Technical Details

The demo uses:
- Vanilla JavaScript for the neural network implementation
- React for the user interface
- SVG for visualizations
- Tailwind CSS for styling

The network uses:
- Mean Squared Error loss function
- Gradient descent for optimization
- Configurable activation functions (ReLU/Sigmoid)
- Random weight initialization
  `;

  return (
    <div className="prose max-w-none px-4 py-8">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default Documentation; 