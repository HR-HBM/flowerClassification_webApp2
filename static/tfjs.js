import { IMAGENET_CLASSES } from "./imagenet_classes.js";

const outputDiv = document.getElementById("output");
let model;

async function loadModel() {
  try {
    outputDiv.textContent = "Loading TF Model...";
    model = await tf.loadLayersModel(
      "/static/model/model.json"
    ); // Update URL
    outputDiv.textContent = "TF Model Loaded.";
  } catch (error) {
    outputDiv.textContent = `Error loading model: ${error}`;
  }
}

loadModel();

// Continue in static/tfjs.js
const imageUpload = document.getElementById("imageUpload");
const imagePreview = document.getElementById("imagePreview");

imageUpload.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.src = e.target.result;
      img.onload = async () => {
        imagePreview.src = img.src;
        imagePreview.style.display = "block";
        const processedImage = await preprocessImage(img);
        makePrediction(processedImage);
      };
    };
    reader.readAsDataURL(file);
  }
});

// Continue in static/tfjs.js
async function preprocessImage(imageElement) {
    try {
      let img = tf.browser.fromPixels(imageElement).toFloat();
      img = tf.image.resizeBilinear(img, [224, 224]);
      const offset = tf.scalar(127.5);
      const normalized = img.sub(offset).div(offset);
      const batched = normalized.reshape([1, 224, 224, 3]);
      return batched;
    } catch (error) {
      outputDiv.textContent = `Error in model prediction: ${error}`;
    }
  }

  // Continue in static/tfjs.js
async function makePrediction(processedImage) {
    try {
      const prediction = model.predict(processedImage);
      const highestPredictionIndex = await tf.argMax(prediction, 1).data();
      const label = IMAGENET_CLASSES[highestPredictionIndex];
      outputDiv.textContent = `Prediction: ${label}`;
    } catch (error) {
      outputDiv.textContent = `Error making prediction: ${error}`;
    }
  }