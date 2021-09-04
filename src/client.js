import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';


// console.log(tf.version);
// tf.setBackend('wasm')
// tf.ready().then(() => {
//     console.log(tf.getBackend());
// });
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

const income_source_1 = tf.tensor1d([100, 200, 300, 150]);
const income_source_2 = tf.tensor1d([50, 70, 30, 20]);
const total_income = tf.add(income_source_1, income_source_2);
console.log(`total income is ${total_income}`);

const var_1 = tf.variable(income_source_1);
