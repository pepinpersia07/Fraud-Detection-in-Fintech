const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const app = express();
app.use(bodyParser.json());

let model;
let scaler;

// Load the model and scaler
async function loadModel() {
  model = await tf.loadLayersModel('file://model.h5');
  const scalerParams = JSON.parse(fs.readFileSync('scaler.json'));
  scaler = {
    mean: scalerParams.mean,
    scale: scalerParams.scale
  };
}

// Apply scaling
function scaleInput(input) {
  return input.map(value => (value - scaler.mean) / scaler.scale);
}

// Endpoint to predict
app.post('/predict', async (req, res) => {
  if (!model) {
    await loadModel();
  }

  const { count, amount } = req.body;

  // Scale the input
  const scaledInput = scaleInput([count, amount]);

  // Make a prediction
  const prediction = model.predict(tf.tensor2d([scaledInput], [1, scaledInput.length])).arraySync()[0][0];
  const result = prediction > 0.5 ? 'High Risk' : 'Low Risk';

  res.json({ prediction: result });
});

app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});
