import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import "regenerator-runtime/runtime";
import * as model from './model';

import $ from "jquery";
import "materialize-css";
import "material-icons";
import "./main.scss";

M.AutoInit();


console.log("line1");
const init = async () => {
    await tf.ready();
    console.log(tf.getBackend());
    model.train();
}
init();
console.log("line4")
//
// const age = tf.tensor1d([30, 25], 'int32');
// age.print();
// tf.print(age);
// console.log(age.shape);
// console.log(age.dtype);
//
// const age_income_height = tf.tensor2d([[30, 1000, 170], [25, 2000, 168]]);
// age_income_height.print();
// console.log(age_income_height.shape);
// console.log(age_income_height.dtype);
//
// const multiplier = tf.scalar(10);
// multiplier.print();
// console.log(multiplier.dtype);


