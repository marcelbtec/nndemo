import React, { useState, useEffect, useCallback } from 'react';
import { Play, Square, RotateCcw } from 'lucide-react';

// Helper functions with numerical stability
const sigmoid = (x) => {
  const clipped = Math.max(-10, Math.min(10, x));
  return 1 / (1 + Math.exp(-clipped));
};

const sigmoid_derivative = (x) => {
  const sig = sigmoid(x);
  return sig * (1 - sig);
};

const dataGenerators = {
  linear: () => {
    const data = [];
    // Increased from 20 to 100 points per class
    for (let i = 0; i < 100; i++) {
      data.push({
        x: -0.5 + Math.random() * 0.4 - 0.2,
        y: -0.5 + Math.random() * 0.4 - 0.2,
        class: 0
      });
    }
    for (let i = 0; i < 100; i++) {
      data.push({
        x: 0.5 + Math.random() * 0.4 - 0.2,
        y: 0.5 + Math.random() * 0.4 - 0.2,
        class: 1
      });
    }
    return data;
  },

  circular: () => {
    const data = [];
    // Increased from 20 to 100 points per class
    for (let i = 0; i < 100; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() * 0.3;
      data.push({
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
        class: 0
      });
    }
    for (let i = 0; i < 100; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const radius = 0.7 + Math.random() * 0.3;
      data.push({
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
        class: 1
      });
    }
    return data;
  },

  xor: () => {
    const data = [];
    // Increased from 10 to 50 points per quadrant
    for (let i = 0; i < 50; i++) {
      data.push({
        x: 0.3 + Math.random() * 0.4,
        y: 0.3 + Math.random() * 0.4,
        class: 1
      });
      data.push({
        x: -0.7 + Math.random() * 0.4,
        y: -0.7 + Math.random() * 0.4,
        class: 1
      });
      data.push({
        x: -0.7 + Math.random() * 0.4,
        y: 0.3 + Math.random() * 0.4,
        class: 0
      });
      data.push({
        x: 0.3 + Math.random() * 0.4,
        y: -0.7 + Math.random() * 0.4,
        class: 0
      });
    }
    return data;
  },

  spiral: () => {
    const data = [];
    // Increased from 20 to 100 points per spiral
    const n = 100;
    for (let i = 0; i < n; i++) {
      const r = i / n * 0.8;
      const t = 1.25 * i / n * 2 * Math.PI;
      
      data.push({
        x: r * Math.cos(t),
        y: r * Math.sin(t),
        class: 0
      });
      
      data.push({
        x: r * Math.cos(t + Math.PI),
        y: r * Math.sin(t + Math.PI),
        class: 1
      });
    }
    return data;
  }
};

// Add these at the top with your other helper functions
const relu = (x) => Math.max(0, x);
const relu_derivative = (x) => x > 0 ? 1 : 0;

class SimpleNN {
  constructor(hiddenUnits = 10, learningRate = 0.1) {
    const initializeWeight = (fanIn) => {
      return () => (Math.random() - 0.5) * Math.sqrt(2.0 / fanIn);
    };

    const initInput = initializeWeight(2);
    const initHidden = initializeWeight(hiddenUnits);

    this.weightsIH = [
      Array(hiddenUnits).fill().map(initInput),
      Array(hiddenUnits).fill().map(initInput)
    ];
    this.biasH = Array(hiddenUnits).fill().map(initInput);
    this.weightsHO = Array(hiddenUnits).fill().map(initHidden);
    this.biasO = initHidden();

    // Add momentum arrays
    this.velocityIH = [
      Array(hiddenUnits).fill().map(() => 0),
      Array(hiddenUnits).fill().map(() => 0)
    ];
    this.velocityHO = Array(hiddenUnits).fill().map(() => 0);
    this.velocityBiasH = Array(hiddenUnits).fill().map(() => 0);
    this.velocityBiasO = 0;

    this.learningRate = learningRate;
    this.momentum = 0.9;  // Momentum coefficient
    this.useRelu = true;  // Default to ReLU
    this.trainingData = [];
    this.hiddenUnits = hiddenUnits;
  }

  activate(x) {
    return this.useRelu ? relu(x) : sigmoid(x);
  }

  activate_derivative(x) {
    return this.useRelu ? relu_derivative(x) : sigmoid_derivative(x);
  }

  forward(input) {
    this.input = input;
    
    this.hiddenPreActivation = Array(this.hiddenUnits).fill(0);
    for (let i = 0; i < this.hiddenUnits; i++) {
      this.hiddenPreActivation[i] = this.biasH[i];
      for (let j = 0; j < 2; j++) {
        this.hiddenPreActivation[i] += this.weightsIH[j][i] * input[j];
      }
    }
    this.hiddenActivation = this.hiddenPreActivation.map(x => this.activate(x));
    
    this.outputPreActivation = this.biasO;
    for (let i = 0; i < this.hiddenUnits; i++) {
      this.outputPreActivation += this.weightsHO[i] * this.hiddenActivation[i];
    }
    this.outputActivation = sigmoid(this.outputPreActivation);  // Always use sigmoid for output
    
    return this.outputActivation;
  }

  backpropagate(target) {
    const outputError = this.outputActivation - target;
    const outputDelta = outputError * sigmoid_derivative(this.outputPreActivation);
    
    const hiddenDeltas = Array(this.hiddenUnits).fill(0);
    for (let i = 0; i < this.hiddenUnits; i++) {
      hiddenDeltas[i] = this.weightsHO[i] * outputDelta * 
        this.activate_derivative(this.hiddenPreActivation[i]);
    }
    
    // Update weights with momentum
    for (let i = 0; i < this.hiddenUnits; i++) {
      const velocityHO = this.momentum * this.velocityHO[i] - 
        this.learningRate * outputDelta * this.hiddenActivation[i];
      this.velocityHO[i] = velocityHO;
      this.weightsHO[i] += velocityHO;
    }
    
    const velocityBiasO = this.momentum * this.velocityBiasO - 
      this.learningRate * outputDelta;
    this.velocityBiasO = velocityBiasO;
    this.biasO += velocityBiasO;
    
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < this.hiddenUnits; j++) {
        const velocityIH = this.momentum * this.velocityIH[i][j] - 
          this.learningRate * hiddenDeltas[j] * this.input[i];
        this.velocityIH[i][j] = velocityIH;
        this.weightsIH[i][j] += velocityIH;
      }
    }
    
    for (let i = 0; i < this.hiddenUnits; i++) {
      const velocityBiasH = this.momentum * this.velocityBiasH[i] - 
        this.learningRate * hiddenDeltas[i];
      this.velocityBiasH[i] = velocityBiasH;
      this.biasH[i] += velocityBiasH;
    }
  }

  train() {
    let totalLoss = 0;
    const shuffled = [...this.trainingData].sort(() => Math.random() - 0.5);
    
    for (const point of shuffled) {
      const input = [point.x, point.y];
      const output = this.forward(input);
      this.backpropagate(point.class);
      
      const error = point.class - output;
      totalLoss += error * error;
    }
    
    return totalLoss / this.trainingData.length;
  }

  predict(x, y) {
    return this.forward([x, y]);
  }

  getWeights() {
    return {
      inputHidden: this.weightsIH.map(row => [...row]),
      hiddenOutput: [...this.weightsHO]
    };
  }

  setTrainingData(data) {
    this.trainingData = data;
  }
}

const DecisionBoundary = ({ network, width, height }) => {
  const gridSize = 20;
  const boundaryPoints = [];
  
  for (let x = 0; x < gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      const dataX = (x / gridSize) * 4 - 2;
      const dataY = (y / gridSize) * 4 - 2;
      
      const prediction = network.predict(dataX, dataY);
      
      boundaryPoints.push({
        x: dataX,
        y: dataY,
        prediction: prediction
      });
    }
  }

  const toSvgX = x => (x + 2) * width/4;
  const toSvgY = y => height - (y + 2) * height/4;

  return (
    <g>
      {boundaryPoints.map((point, i) => {
        const cellWidth = width / gridSize;
        const cellHeight = height / gridSize;
        
        const redComponent = Math.floor(point.prediction * 255);
        const blueComponent = Math.floor((1 - point.prediction) * 255);
        
        return (
          <rect
            key={i}
            x={toSvgX(point.x) - cellWidth/2}
            y={toSvgY(point.y) - cellHeight/2}
            width={cellWidth}
            height={cellHeight}
            fill={`rgb(${redComponent}, 0, ${blueComponent})`}
            opacity={0.3}
          />
        );
      })}
    </g>
  );
};

const NetworkViz = () => {
  const [hiddenUnits, setHiddenUnits] = useState(10);
  const [learningRate, setLearningRate] = useState(0.1);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [dataPattern, setDataPattern] = useState('linear');
  const [weights, setWeights] = useState({
    inputHidden: [
      Array(10).fill().map(() => 0.1),  // Start with 10 units
      Array(10).fill().map(() => 0.1)
    ],
    hiddenOutput: Array(10).fill().map(() => 0.1)
  });
  const [loss, setLoss] = useState(0);
  const [network, setNetwork] = useState(() => {
    const nn = new SimpleNN(10, 0.1);  // Use fixed initial values
    nn.setTrainingData(dataGenerators.linear());
    return nn;
  });
  const [trainingData, setTrainingData] = useState(() => dataGenerators.linear());


  const trainStep = useCallback(() => {
    const currentLoss = network.train();
    const newWeights = network.getWeights();
    
    setEpoch(e => e + 1);
    setWeights(newWeights);
    setLoss(currentLoss);
  }, [network]);

  useEffect(() => {
    if (!isTraining) return;
    
    const interval = setInterval(trainStep, 100);
    return () => clearInterval(interval);
  }, [isTraining, trainStep]);

  const handleReset = useCallback(() => {
    setIsTraining(false);
    setEpoch(0);
    const nn = new SimpleNN(hiddenUnits, learningRate);
    const newData = dataGenerators[dataPattern]();
    nn.setTrainingData(newData);
    setNetwork(nn);  // Make sure to update the network state
    setTrainingData(newData);
    setWeights({
      inputHidden: [
        Array(hiddenUnits).fill().map(() => 0.1),
        Array(hiddenUnits).fill().map(() => 0.1)
      ],
      hiddenOutput: Array(hiddenUnits).fill().map(() => 0.1)
    });
    setLoss(0);
}, [dataPattern, hiddenUnits, learningRate]);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6 bg-gray-900 text-gray-100">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white">Binary Classification with Neural Network</h1>
        <div className="flex items-center space-x-4">
          <select
            value={dataPattern}
            onChange={(e) => {
              setDataPattern(e.target.value);
              setIsTraining(false);
              const newData = dataGenerators[e.target.value]();
              const nn = new SimpleNN(hiddenUnits, learningRate);
              nn.setTrainingData(newData);
              setNetwork(nn);
              setTrainingData(newData);
              setEpoch(0);
              setLoss(0);
            }}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-gray-100"
          >
            <option value="linear">Linear</option>
            <option value="circular">Circular</option>
            <option value="xor">XOR</option>
            <option value="spiral">Spiral</option>
          </select>
          <button
            onClick={() => setIsTraining(!isTraining)}
            className="inline-flex items-center px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 transition-colors"
          >
            {isTraining ? <Square className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
            {isTraining ? 'Stop' : 'Train'}
          </button>
          <button
            onClick={handleReset}
            className="inline-flex items-center px-4 py-2 rounded bg-gray-700 text-white hover:bg-gray-600 transition-colors"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </button>
        </div>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
        <h2 className="text-lg font-semibold mb-4 text-white">Network Parameters</h2>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300">
              Hidden Units: {hiddenUnits}
              <input
                type="range"
                min="4"
                max="20"
                value={hiddenUnits}
                onChange={(e) => {
                  const newUnits = parseInt(e.target.value);
                  setHiddenUnits(newUnits);
                  setIsTraining(false);
                  const nn = new SimpleNN(newUnits, learningRate);
                  const newData = dataGenerators[dataPattern]();
                  nn.setTrainingData(newData);
                  setNetwork(nn);
                  setTrainingData(newData);
                  setWeights({
                    inputHidden: [
                      Array(newUnits).fill().map(() => 0.1),
                      Array(newUnits).fill().map(() => 0.1)
                    ],
                    hiddenOutput: Array(newUnits).fill().map(() => 0.1)
                  });
                  setEpoch(0);
                  setLoss(0);
                }}
                className="w-full mt-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </label>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300">
              Learning Rate: {learningRate.toFixed(3)}
              <input
                type="range"
                min="0.001"
                max="0.5"
                step="0.001"
                value={learningRate}
                onChange={(e) => {
                  const newRate = parseFloat(e.target.value);
                  setLearningRate(newRate);
                  network.learningRate = newRate;
                }}
                className="w-full mt-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </label>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300">
              Activation Function
              <select
                value={network.useRelu ? 'relu' : 'sigmoid'}
                onChange={(e) => {
                  const nn = new SimpleNN(hiddenUnits, learningRate);
                  nn.useRelu = e.target.value === 'relu';
                  const newData = dataGenerators[dataPattern]();
                  nn.setTrainingData(newData);
                  setNetwork(nn);
                  setTrainingData(newData);
                  setEpoch(0);
                  setLoss(0);
                  setIsTraining(false);
                }}
                className="w-full mt-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100"
              >
                <option value="relu">ReLU</option>
                <option value="sigmoid">Sigmoid</option>
              </select>
            </label>
          </div>
        </div>
        
        <div className="mt-4 text-sm text-gray-400">
          <p>Recommended settings for different patterns:</p>
          <ul className="list-disc pl-5 mt-2">
            <li>Linear: 4-6 hidden units, learning rate 0.1, Sigmoid</li>
            <li>Circular: 10-15 hidden units, learning rate 0.1, ReLU</li>
            <li>XOR: 8-10 hidden units, learning rate 0.1, ReLU</li>
            <li>Spiral: 15-20 hidden units, learning rate 0.05, ReLU</li>
          </ul>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
          <h2 className="text-lg font-semibold mb-4 text-white">Classification Problem</h2>
          <div className="mb-4 text-sm text-gray-400">
            <p>Binary classification of two clusters:</p>
            <ul className="list-disc pl-5 mt-2">
              <li>Class 0 (Blue): Centered around (-0.5, -0.5)</li>
              <li>Class 1 (Red): Centered around (0.5, 0.5)</li>
            </ul>
            <p className="mt-2">Task: Learn to separate the two classes</p>
          </div>
          <svg width="300" height="300" className="border border-gray-700">
            <DecisionBoundary 
              network={network}
              width={300}
              height={300}
            />
            <line x1="0" y1="150" x2="300" y2="150" stroke="#4B5563" strokeWidth="1" opacity="0.5" />
            <line x1="150" y1="0" x2="150" y2="300" stroke="#4B5563" strokeWidth="1" opacity="0.5" />
            
            {trainingData.map((point, i) => (
              <circle
                key={i}
                cx={150 + point.x * 150}
                cy={150 - point.y * 150}
                r="5"
                fill={point.class === 0 ? '#3b82f6' : '#ef4444'}
              />
            ))}
          </svg>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
          <h2 className="text-lg font-semibold mb-4 text-white">Neural Network</h2>
          <svg width="400" height="300">
            {/* Input Layer */}
            <g transform="translate(50,100)">
              <circle r="20" fill="#1F2937" stroke="#4B5563" />
              <text textAnchor="middle" dy=".3em" fill="#E5E7EB">x₁</text>
            </g>
            <g transform="translate(50,200)">
              <circle r="20" fill="#1F2937" stroke="#4B5563" />
              <text textAnchor="middle" dy=".3em" fill="#E5E7EB">x₂</text>
            </g>

            {/* Hidden Layer */}
            {Array(hiddenUnits).fill(0).map((_, i) => {
              const totalHeight = 280;
              const spacing = totalHeight / (hiddenUnits + 1);
              const yPos = 20 + spacing * (i + 1);
              
              return (
                <g key={i} transform={`translate(200,${yPos})`}>
                  <circle r="20" fill="#1F2937" stroke="#4B5563" />
                  <text textAnchor="middle" dy=".3em" fill="#E5E7EB">h{i+1}</text>
                </g>
              );
            })}

            {/* Output Layer */}
            <g transform="translate(350,150)">
              <circle r="20" fill="#1F2937" stroke="#4B5563" />
              <text textAnchor="middle" dy=".3em" fill="#E5E7EB">y</text>
            </g>

            {/* Connections */}
            {weights.inputHidden.map((row, i) => 
              row.map((weight, j) => {
                const totalHeight = 280;
                const spacing = totalHeight / (hiddenUnits + 1);
                const hiddenY = 20 + spacing * (j + 1);

                return (
                  <line 
                    key={`ih-${i}-${j}`}
                    x1="70" 
                    y1={100 + i * 100}
                    x2="180" 
                    y2={hiddenY}
                    stroke={weight > 0 ? '#3B82F6' : '#EF4444'}
                    strokeWidth={Math.abs(weight) * 3}
                    opacity={0.5}
                  />
                );
              })
            )}

            {weights.hiddenOutput.map((weight, i) => {
              const totalHeight = 280;
              const spacing = totalHeight / (hiddenUnits + 1);
              const hiddenY = 20 + spacing * (i + 1);

              return (
                <line 
                  key={`ho-${i}`}
                  x1="220" 
                  y1={hiddenY}
                  x2="330" 
                  y2="150"
                  stroke={weight > 0 ? '#3B82F6' : '#EF4444'}
                  strokeWidth={Math.abs(weight) * 3}
                  opacity={0.5}
                />
              );
            })}
          </svg>
        </div>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="font-semibold mb-2 text-white">Training Progress</h3>
            <div className="text-gray-300">Epoch: {epoch}</div>
            <div className="text-gray-300">Loss: {loss.toFixed(4)}</div>
            <div className="text-gray-300">Status: {isTraining ? 'Training' : 'Stopped'}</div>
          </div>
          <div>
            <h3 className="font-semibold mb-2 text-white">Network Weights</h3>
            <div className="text-sm text-gray-400">
              <div>Input→Hidden:</div>
              {weights.inputHidden.map((row, i) => (
                <div key={i}>
                  Input {i+1}: [{row.map(w => w.toFixed(2)).join(', ')}]
                </div>
              ))}
              <div>Hidden→Output:</div>
              <div>[{weights.hiddenOutput.map(w => w.toFixed(2)).join(', ')}]</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetworkViz;

