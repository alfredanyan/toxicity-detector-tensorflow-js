import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
export const train = () => {
  tf.tidy(() => {
    run();
  });
};

const csvUrl = "data/toxic_data_sample.csv";
const stopwords = [
  "i",
  "me",
  "my",
  "myself",
  "we",
  "our",
  "ours",
  "ourselves",
  "you",
  "your",
  "yours",
  "yourself",
  "yourselves",
  "he",
  "him",
  "his",
  "himself",
  "she",
  "her",
  "hers",
  "herself",
  "it",
  "its",
  "itself",
  "they",
  "them",
  "their",
  "theirs",
  "themselves",
  "what",
  "which",
  "who",
  "whom",
  "this",
  "that",
  "these",
  "those",
  "am",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "having",
  "do",
  "does",
  "did",
  "doing",
  "a",
  "an",
  "the",
  "and",
  "but",
  "if",
  "or",
  "because",
  "as",
  "until",
  "while",
  "of",
  "at",
  "by",
  "for",
  "with",
  "about",
  "against",
  "between",
  "into",
  "through",
  "during",
  "before",
  "after",
  "above",
  "below",
  "to",
  "from",
  "up",
  "down",
  "in",
  "out",
  "on",
  "off",
  "over",
  "under",
  "again",
  "further",
  "then",
  "once",
  "here",
  "there",
  "when",
  "where",
  "why",
  "how",
  "all",
  "any",
  "both",
  "each",
  "few",
  "more",
  "most",
  "other",
  "some",
  "such",
  "no",
  "nor",
  "not",
  "only",
  "own",
  "same",
  "so",
  "than",
  "too",
  "very",
  "s",
  "t",
  "can",
  "will",
  "just",
  "don",
  "should",
  "now",
];
let tmpDictionary = {};
let EMBEDDING_SIZE = 1000;
const BATCH_SIZE = 16;
const render = true;
const TRAINING_EPOCHS = 10;

const MODEL_ID = 'toxicity-detector-tfidf';
const IDF_STORAGE_ID = 'toxicity-idfs';
const DICTIONARY_STORAGE_ID = 'toxicity-tfidf-dictionary';
//HELPER FUNCTIONS

const getInverseDocumentFrequency = (documentTokens, dictionary) => {
  return dictionary.map(
    (token) =>
      1 +
      Math.log(
        documentTokens.length /
          documentTokens.reduce(
            (acc, curr) => (curr.includes(token) ? acc + 1 : acc),
            0
          )
      )
  );
};

const encoder = (sentence, dictionary, idfs) => {
  const tokens = tokenize(sentence);
  const tfs = getTermFrequency(tokens, dictionary);
  const tfidfs = getTfIdf(tfs, idfs);
  return tfidfs;
};

const getTfIdf = (tfs, idfs) => {
  return tfs.map((element, index) => element * idfs[index]);
};
const getTermFrequency = (tokens, dictionary) => {
  return dictionary.map((token) =>
    tokens.reduce((acc, curr) => (curr == token ? acc + 1 : acc), 0)
  );
};

const readRawData = () => {
  const readData = tf.data.csv(csvUrl, {
    columnConfigs: {
      toxic: {
        isLabel: true,
      },
    },
  });
  return readData;
};
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
      value: labelCounts[key],
    });
  });
  // console.log(barChartData);
  tfvis.render.barchart(
    {
      tab: "Exploration",
      name: "Toxic output labels",
    },
    barChartData
  );
};
const tokenize = (sentence, isCreateDict = false) => {
  const tmpTokens = sentence.split(/\s+/g);
  const tokens = tmpTokens.filter(
    (token) => !stopwords.includes(token) && token.length > 0
  );

  if (isCreateDict) {
    const labelCounts = tokens.reduce((acc, token) => {
      acc[token] = acc[token] === undefined ? 1 : (acc[token] += 1);
      return acc;
    }, tmpDictionary);
  }
  return tmpTokens;
};

const sortDictionaryByValue = (dict) => {
  const items = Object.keys(dict).map((key) => {
    return [key, dict[key]];
  });
  return items.sort((first, second) => {
    return second[1] - first[1];
  });
};

//load data
const prepareData = (dictionary, idfs) => {
  const preprocess = ({ xs, ys }) => {
    const comment = xs["comment_text"];
    const trimmedComment = comment.toLowerCase().trim();
    const encoded = encoder(trimmedComment, dictionary, idfs);
    return {
      xs: tf.tensor2d([encoded], [1, dictionary.length]),
      ys: tf.tensor2d([ys["toxic"]], [1, 1]),
    };
  };

  const readData = tf.data
    .csv(csvUrl, {
      columnConfigs: {
        toxic: {
          isLabel: true,
        },
      },
    })
    .map(preprocess);
  return readData;
};

const sampleTest = () => {
    const documentTokens = [];
    const testComments = ["I loved the movie", "movie was boring"];
    testComments.forEach((row) => {
      const comment = row.toLowerCase();
      documentTokens.push(tokenize(comment, true));
    });
    console.log(tmpDictionary);
    const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary);
    const dictionary = sortedTmpDictionary.map((row) => row[0]);
    const idfs = getInverseDocumentFrequency(documentTokens, dictionary);
  
    testComments.forEach((row) => {
      const comment = row.toLowerCase();
      console.log(encoder(comment, dictionary, idfs));
    });
  };
  

const prepareDataUsingGenerator = (comments, labels, dictionary, idfs) => {
  function* getFeatures() {
    for (let i = 0; i < comments.length; i++) {
      //generate one sample at a time
      const encoded = encoder(comments[i], dictionary, idfs);
      yield tf.tensor2d([encoded], [1, dictionary.length]);
    }
  }
  function* getLabels() {
    for (let i = 0; i < labels.length; i++) {
      yield tf.tensor2d([labels[i]], [1, 1]);
    }
  }
  const xs = tf.data.generator(getFeatures);
  const ys = tf.data.generator(getLabels);
  const ds = tf.data.zip({ xs, ys });
  return ds;
};

const trainTestSplit = (dataSet, nrows) => {
  const trainValidationCount = Math.round(nrows * 0.7);
  const trainingCount = Math.round(nrows * 0.6)
  const SEED = 7687547;

  const trainValidationData = dataSet
    .shuffle(nrows, SEED)
    .take(trainValidationCount);

  const testDataset = dataSet
    .shuffle(nrows, SEED)
    .skip(trainValidationCount)
    .batch(BATCH_SIZE);

  const trainingDataset = trainValidationData
    .take(trainingCount)
    .batch(BATCH_SIZE);

  const validationDataset = trainValidationData
    .skip(trainingCount)
    .take(nrows * 0.1);

    return {
        trainingDataset,
        validationDataset,
        testDataset
    };
};

const buildModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [EMBEDDING_SIZE],
        activation: "relu",
        units: 5
    }));
    model.add(tf.layers.dense({
        activation: "sigmoid",
        units: 1
    }));
    model.compile({
        loss: "binaryCrossentropy",
        optimizer: tf.train.adam(0.06),
        metrics: ["accuracy"]
    });
    model.summary();
    return model;
    
}



const earlyStoppingCallback = tf.callbacks.earlyStopping({
    monitor: 'val_acc',
    minDelta: 0.3,
    patience: 5,
    verbose: 1
}); 



const trainModel = async (model, trainingDataset, validationDataset) => {
    const history = [];
    const surface = {
        name: 'onEpochEnd Performance ', tab: 'Training' };

    const batchHistory = [];
    const batchSurface = { name: 'onBatchEnd Performance', tab: 'Training' };
    const messageCallback = new tf.CustomCallback(
        {
            onEpochEnd: async (epoch, logs) => {
                history.push(logs);
                console.log("Epoch: " + epoch + " loss: " + logs.loss);
                if (render) {
                    tfvis.show.history(surface, history, ['loss', 'val_loss', 'acc', 'val_acc']); 
                }
            },
            onBatchEnd: async (batch, logs) => {
                batchHistory.push(logs);
                if (render) {
                    tfvis.show.history(batchSurface, batchHistory, ['loss', 'val_loss', 'acc', 'val_acc'])
                }
            } 
        }
    );
    
    
    const trainResult = await model.fitDataset(trainingDataset, 
        {
            epochs: TRAINING_EPOCHS,
            validationData: validationDataset,

            callbacks:  [
                messageCallback,
                // earlyStoppingCallback 
            ]
        }
    );
        return model;
}

const evaluateModel = async (model, testDataset) => {
    const modelResult = await model.evaluateDataset(testDataset);
    const testLoss = modelResult[0].dataSync()[0];
    const testAcc = modelResult[1].dataSync()[0];

    console.log(`Loss on Test Dataset : ${testLoss.toFixed(4)}`);
    console.log(`Accuracy on Test Dataset: ${testAcc.toFixed(4)}`);
}

const getMoreEvaluationSummaries = async (model, testDataset) => {
    const allActualLabels = [];
    const allPredictedLabels = [];

    await testDataset.forEachAsync((row) => {
        const actualLabels = row['ys'].dataSync();
        actualLabels.forEach((x) => allActualLabels.push(x));
    
        const features = row['xs'];
        const prediction = model.predictOnBatch(tf.squeeze(features, 1));
        const predictedLabels = tf.round(prediction).dataSync();
        predictedLabels.forEach((x) => allPredictedLabels.push(x));
    
    });

    const allActualLabelsTensor = tf.tensor1d(allActualLabels);
    const allPredictedLabelsTensors = tf.tensor1d(allPredictedLabels);

    const accuracyResult = await tfvis.metrics.accuracy(allActualLabelsTensor, allPredictedLabelsTensors);
    console.log(`Accuracy result: ${accuracyResult}` )

    //calc per class accuracy
    const perClassAccuracyResult = await tfvis.metrics.perClassAccuracy(allActualLabelsTensor, allPredictedLabelsTensors);
    console.log(`Per Class Accuracy result: ${JSON.stringify(perClassAccuracyResult, null, 2)}`);

    const confusionMatrixResult = await tfvis.metrics.confusionMatrix(allActualLabelsTensor, allPredictedLabelsTensors);
    const confusionMatrixVizResult = { "values": confusionMatrixResult };

    console.log(`confusion matrix : \n ${JSON.stringify(confusionMatrixVizResult, null, 2)}`)

    const surface = { tab: 'Evaluation', name: 'Confusion Matrix'};
    if (render) {
        tfvis.render.confusionMatrix(surface, confusionMatrixVizResult);
    }


}

const exportModel = async (model, modelID, idfs, idfsStorageID, dictionary, dictionaryStorageID) => {
    // const modelPath = `downloads://${modelID}`
    const modelPath = `localstorage://${modelID}`;
    const saveModelResults = await model.save(modelPath);
    console.log('model exported');

    localStorage.setItem(dictionaryStorageID, JSON.stringify(dictionary));
    localStorage.setItem(idfsStorageID, JSON.stringify(idfs));
    console.log('dictionary and IDFs exported');
    return saveModelResults;
}

const loadModel = async () => {
    const modelPath = `localstorage://${MODEL_ID}`;
    console.log(modelPath);

    const models = await tf.io.listModels();
    if (models[modelPath]) {
        console.log('model exists');
        const model_loaded = await tf.loadLayersModel(modelPath);
        return model_loaded;
    } else {
        console.log('no model available');
        return null;
    }
}

const predictResults = async (sentence, model) => {
    console.log('prediction started');

    const idfObject = localStorage.getItem(IDF_STORAGE_ID);
    const idfs = JSON.parse(idfObject);

    const dictionaryObject = localStorage.getItem(DICTIONARY_STORAGE_ID);
    const dictionary = JSON.parse(dictionaryObject);

    //get encoded values
    const encoded = encoder(sentence.toLowerCase().trim(), dictionary, idfs);

    const encodedTensor = tf.tensor2d([encoded], [1, dictionary.length]);


    //make predictions
    const predictionResult = model.predict(encodedTensor);
    const predictionScore = predictionResult.dataSync();

    //extract prediction class
    const predictedClass = (predictionScore >= 0.5) ? "toxic" : "non-toxic";
    const resultMessage = `Probability : ${predictionScore}, Class: ${predictedClass}`;
    console.log(resultMessage);
    return predictedClass;
}

const run = async () => {
  const rawDataResult = readRawData();
  const labels = [];
  const comments = [];
  const documentTokens = [];

  await rawDataResult.forEachAsync((row) => {
    // console.log(row);
    const comment = row["xs"]["comment_text"];
    const trimmedComment = comment.toLowerCase().trim();
    comments.push(trimmedComment);
    documentTokens.push(tokenize(trimmedComment, true));
    labels.push(row["ys"]["toxic"]);
  });

  //plot labels
  if (render) {
      plotOutputLabelCounts(labels);
  }

  // console.log(Object.keys(tmpDictionary).length);
  // console.log(tmpDictionary);
  const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary);
  // console.log( sortedTmpDictionary);
  if (sortedTmpDictionary.length <= EMBEDDING_SIZE) {
    EMBEDDING_SIZE = sortedTmpDictionary.length;
  }

  const dictionary = sortedTmpDictionary
    .slice(0, EMBEDDING_SIZE)
    .map((row) => row[0]);
  // console.log(dictionary);

  //calculate idf
  const idfs = getInverseDocumentFrequency(documentTokens, dictionary);

  //preprocessing data
  // const dataSet = prepareData(dictionary, idfs);
  // await dataSet.forEachAsync((e) => console.log(e));

  // sampleTest();

  //preprocessing data - approach 2 for large datasets

  const dataSet = prepareDataUsingGenerator(comments, labels, dictionary, idfs);
//   await dataSet.forEachAsync((e) => console.log(e));

  //splitting data sets
  const { trainingDataset, validationDataset, testDataset } = trainTestSplit(dataSet, documentTokens.length);
//   await trainingDataset.forEachAsync((e) => console.log(e));

    let model =  buildModel();

    if (render) {
        tfvis.show.modelSummary({
            name: 'Model Summary',
            tab: 'Model'},
            model);
    }

    model = await trainModel(model, trainingDataset, validationDataset);

    await evaluateModel(model,  testDataset);
    await getMoreEvaluationSummaries(model, testDataset)

    // //export model
    const exportResult = await exportModel(model, MODEL_ID, idfs, IDF_STORAGE_ID, dictionary, DICTIONARY_STORAGE_ID);

    //load model
    // const model_loaded = await loadModel();
    // model_loaded.summary();

    // //predict using trained model
    // const example_1 = 'You are doing great';
    // predictResults(example_1, model_loaded);

    // const example_2 = 'You are fucked up';
    // predictResults(example_2, model_loaded);
};






