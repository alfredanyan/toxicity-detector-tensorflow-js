import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
export const train = () => {
    tf.tidy(() => {
        run();
    });
};
const csvUrl = 'data/toxic_data_sample.csv';
const readRawData = () => {

    const readData = tf.data.csv(csvUrl, {
        columnConfigs: {
            toxic: {
                isLabel: true
            }
        }
    });
    return readData;
}
const plotOutputLabelCounts = (labels) => {
    const labelCounts = labels.reduce((acc, label) => {
        acc[label] = acc[label] === undefined ? 1 : acc[label] + 1;
        return acc;
    }, {});
    console.log(labelCounts);
}

const run = async () => {
    const rawDataResult = readRawData();
    const labels = [];
    await rawDataResult.forEachAsync((row) => {
        // console.log(row);
        labels.push(row['ys']['toxic']);
    })

    //plot labels
    plotOutputLabelCounts(labels);
}
