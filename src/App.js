import React from 'react';
import NeuralNetworkViz from './NeuralNetworkViz';

function App() {
  return (
    <div className="App">
      <div className="absolute top-4 right-4">
        <a 
          href="/docs/index.html" 
          target="_blank" 
          rel="noopener noreferrer"
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Documentation
        </a>
      </div>
      <NeuralNetworkViz />
    </div>
  );
}

export default App;