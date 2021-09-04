const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const express = require('express');
const app = express();

app.get('/train', function(req,res) {
    console.log(tf.version);
    tf.ready().then(() => {
        const message = "Loaded TensorFlow.js version" + " " +
            tf.version.tfjs + " \n with backend " + tf.getBackend();
        console.log(message);
        res.send(message);
    })
})

app.listen(9000, function (req,res) {
    console.log('Running server on port 9000...');
})
