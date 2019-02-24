/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
// import * as tfn from '@tensorflow/tfjs-node';


import * as ui from './ui';
import {Webcam} from './webcam';

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));
const LOCAL_MODEL_JSON_URL = 'http://storage.googleapis.com/modelinsight/tfjs/model.json';
const img_sz = 224;
var predictions = 0


let mobileUnet;

// Loads le model
async function loadMobileUnet() {
  const mobileUnet= await tf.loadLayersModel(LOCAL_MODEL_JSON_URL);
  return mobileUnet
}


let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    const output = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();

      predictions = mobileUnet.predict(img);
      const img_clone = tf.reshape( img, [img_sz, img_sz, 3]);
      const pred_mask = tf.reshape(predictions, [img_sz, img_sz, 1]);

      const combo = tf.mul(img_clone, pred_mask);
      return [pred_mask, combo]
    });

    ui.setPredictedComboClass(output[1]); //340sec
    await tf.nextFrame();
  }
  ui.donePredicting();
}


document.getElementById("predict").addEventListener("click", () => {
  isPredicting = true;
  predict();
});

document.getElementById('show_original').addEventListener("click", () => {
  isPredicting = false;
  ui.clearPredictedMaskClass();
});


async function init() {
  try {
    await webcam.setup();
  } catch (e) {
    document.getElementById('no-webcam').style.display = 'block';
  }
  mobileUnet = await loadMobileUnet();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  tf.tidy(() => mobileUnet.predict(webcam.capture()));

  ui.init();
}

// Initialize the application.
init();
