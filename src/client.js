import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import "regenerator-runtime/runtime";
import * as model from './model';

import $ from "jquery";
import "materialize-css";
import "material-icons";
import "./main.scss";

M.AutoInit();


const init = async () => {
    await tf.ready();
    const init_message = "Powered by TensorFlow.js - version: " + tf.version.tfjs + " with backend: " + tf.getBackend();
    $('#init').text(init_message);
    // console.log(tf.getBackend());
    // model.train();
}
init();

var modelOptionSelect = $('select');
let modelOption = 1;
modelOptionSelect.on('change', (e) => {
    modelOption = parseInt(e.target.value);
});

$('#btn_train').on('click', async () => {
    switch (modelOption) {
        case 1:
            $('#btn_train').addClass("disabled");
            M.toast({ html: 'Training Started'})
            console.log("Training custom model with TFIDF features");
            await model.train();
            break;
        default:
            break; 
    }
})


