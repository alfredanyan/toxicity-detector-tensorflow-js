import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
export const train = () => {
    tf.tidy(() => {
        run();
    });
};
const csvUrl = 'data/toxic_data_sample.csv';
const stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
let tmpDictionary = {};
let EMBEDDING_SIZE = 1000;



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
    // console.log(labelCounts);
    const barChartData = [];
    Object.keys(labelCounts).forEach((key) => {
        barChartData.push({
            index: key,
            value: labelCounts[key]
        });
    });
    // console.log(barChartData);
    tfvis.render.barchart({
        tab: 'Exploration',
        name: 'Toxic output labels'
    }, barChartData);
}
const tokenize = (sentence, isCreateDict = false) => {
    const tmpTokens = sentence.split(/\s+/g);
    const tokens = tmpTokens.filter((token) => !stopwords.includes(token) && token.length > 0);


    if (isCreateDict) {
        const labelCounts = tokens.reduce((acc, token) => {
            acc[token] = acc[token] === undefined ? 1 : acc[token] +=1;
            return acc;
        }, tmpDictionary);
    }
    return tmpTokens;
}

const sortDictionaryByValue = (dict) => {
    const items = Object.keys(dict).map((key) => {
        return [key, dict[key]];
    });
    return items.sort((first, second) => {
        return second[1] - first[1];
    });
}
const run = async () => {
    const rawDataResult = readRawData();
    const labels = [];
    const comments = [];
    const documentTokens = [];

    await rawDataResult.forEachAsync((row) => {
        // console.log(row);
        const comment = row['xs']['comment_text'];
        const trimmedComment = comment.toLowerCase().trim();
        comments.push(trimmedComment);
        documentTokens.push(tokenize(trimmedComment, true));
        labels.push(row['ys']['toxic']);
    })

    //plot labels
    plotOutputLabelCounts(labels);

    console.log(Object.keys(tmpDictionary).length);
    console.log(tmpDictionary);
    const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary);
    console.log(sortedTmpDictionary);
    const dictionary = sortedTmpDictionary.slice(0, EMBEDDING_SIZE).map((row) => row[0]);
}
