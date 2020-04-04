class Connection {
    constructor(weight, delta) {
        this.weight = weight || Math.random();
        this.delta = delta || 0;
    }

    get_weight() {
      return this.weight;
    }

    set_weight(weight) {
      return this.weight = weight;
    }

    get_delta() {
        return this.delta;
    }
}

class Neuron {
    constructor(numOuputs = 0, neuronIndex, weights = []) {
        // this.eta = 0.15; // overall net learning rate
        // this.apha = 0.5; // momentum, multiplier of last deltaWeight , [0.0..n]

        this.eta = 0.02; // overall net learning rate
        this.apha = 0.009; // momentum, multiplier of last deltaWeight , [0.0..n]

        this.outputWeights = [];
        this.outputVal = null;
        this.gradient = null;

        this.neuronIndex = neuronIndex;
        for (let c = 0; c < numOuputs; c++) {

            var weight = weights[c] || null;

            this.outputWeights.push(new Connection(weight));
        }
    }

    setOutputVal(outputVal) {
        this.outputVal = outputVal;
    }

    getOutputVal() {
        return this.outputVal;
    }

    getNeuronIndex() {
        return this.neuronIndex;
    }

    updateInputWeights(prevLayer) {
        // The weights to be updated are in the Connection container
        // in the neurons in the preceding layer

        for (let n = 0; n < prevLayer.length-1; n++) {
            let neuron = prevLayer[n];
            let oldDeltaWeight = neuron.outputWeights[this.neuronIndex];

            let newDeltaWeight = this.eta * neuron.getOutputVal() * this.gradient
                + this.apha * oldDeltaWeight.get_delta();

            neuron.outputWeights[this.neuronIndex].deltaWeight = newDeltaWeight;
            neuron.outputWeights[this.neuronIndex].weight += newDeltaWeight;
        }
    }

    sumDOW(nextLayer) {
        var sum = 0.0;
        for (let n = 0; n < nextLayer.length - 1; n++) {
            sum += this.outputWeights[n].weight * nextLayer[n].gradient;
        }

        return sum;
    }

    calcHiddenGradients(nextLayer) {
        let dow = this.sumDOW(nextLayer);
        this.gradient = dow * this.transferFunctionDerivative(this.outputVal);
    }

    calcOutputGradients(targetVal) {
        let delta = targetVal - this.outputVal;
        this.gradient = delta * this.transferFunctionDerivative(this.outputVal);
    }

    feedForward(prevLayer) {
        var sum = 0.0;

        // Sum the prev layer's output (which are out inputs)
        // Include the bias node from previous layer
        // output = fx(sum(IiWi))
        // console.log('prev layer length = ' + prevLayer.length)
        for (let neuronNum = 0; neuronNum < prevLayer.length; neuronNum++) {
            var neuron = prevLayer[neuronNum];
            let connection = neuron.outputWeights[this.getNeuronIndex()]; // connection to current neuron
            // console.log(connection, 'neuron Index = ' + this.getNeuronIndex());
            sum += neuron.getOutputVal() * connection.get_weight()
        }

        this.outputVal = this.transferFunction(sum);
    }

    transferFunction(x) {
        return Math.tanh(x);
    }

    transferFunctionDerivative(x) {
        return 1.0 - x*x;
    }
}

class Net {
    /**
     *
     * @param {Array} topology neurons per layer
     * for example [2,2,1] - 2 input neurons, 2 hidden neurons, 1 ouput
     */
    constructor(topology = []) {
        this.topology = topology;
        this.error = 0.9;
        this.recentAvarageError = 0.0;
        this.recentAverageSmoothingFactor = 0.0;

        this.allNeurons = [];

        // number of layers
        this.numLayers = topology.length;
        this.layers = [];
        for (let layerNum = 0; layerNum < this.numLayers; layerNum++) {
            var layer = [];
            var numOuputs = topology[layerNum + 1] ? topology[layerNum + 1] : 0;
            for (let neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
                let neuron = new Neuron(numOuputs, neuronNum);
                this.allNeurons.push(neuron)
                layer.push(neuron);
            }
            this.layers.push(layer);
        }

    }

    getResults() {
        this.resultVals = [];

        var lastLayer = this.layers[this.layers.length-1];
        //console.log(lastLayer);
        for (let n = 0; n < lastLayer.length-1; n++) {
            this.resultVals.push(lastLayer[n].getOutputVal());
        }

        return this.resultVals;
    }

    getNeurons() {
        return this.allNeurons;
    }

    setNeurons(savedNeurons = []) {
        for (let i = 0; i < savedNeurons.length; i++) {
            const neuron = savedNeurons[i];
            // this.allNeurons[i].setWeights();
        }
        // return this.allNeurons;
    }

    feedForward(inputVals) {
        if(inputVals.length !== this.layers[0].length -1) {
            throw new Exception('Inputs array size does not mismatched input layer size of neuron network')
        }

        // set input values to input layer of neurons
        for(let i=0; i < inputVals.length; i++) {
            this.layers[0][i].setOutputVal(inputVals[i]);
        }

        // Forward propogate starts from second neurons layer
        for(var layerNum = 1; layerNum < this.layers.length; layerNum ++) {
            for (let neuronNum = 0; neuronNum < this.layers[layerNum].length-1; neuronNum++) {

                // let inputVals = this.layers[layerNum-1].map(neuron => neuron.getOutputVal());
                let prevLayer = this.layers[layerNum-1];
                this.layers[layerNum][neuronNum].feedForward(prevLayer);
            }
        }

        // var res = JSON.stringify(this.layers);
        // console.log(JSON.parse(res));
    }

    backProp(targetVals = []) {
        // Calculate overall net error (RMS of of output neuron errors)

        var outputLayer = this.layers[this.layers.length-1];
        this.error = 0.0;

        for (let n = 0; n < outputLayer.length - 1; n++) {
            let delta = targetVals[n] - outputLayer[n].getOutputVal();
            this.error += delta*delta;

        }

        this.error = this.error/(outputLayer.length - 1); //get error squired
        this.error = Math.sqrt(this.error); //RMS

        //console.log(this.error);

        // Implement a recent average measurement
        this.recentAvarageError =
            (this.recentAverageError * this.recentAverageSmoothingFactor + this.error)
            /(this.recentAverageSmoothingFactor + 1.0)

        // Calculate output layer gradients(delta)
        for (let n = 0; n < outputLayer.length -1; n++) {
            let neuron = outputLayer[n];
            neuron.calcOutputGradients(targetVals[n]);
        }

        // console.log(outputLayer);

        // Calculate gradients on hidden layers
        for (var layerNum = (this.layers.length - 2); layerNum > 0; layerNum--) {

            var hiddenLayer = this.layers[layerNum];
            var nextLayer = this.layers[layerNum + 1];

            // console.log('hidden layer ' + layerNum + '=================');
            // console.log(hiddenLayer);
            // console.log('next layer ' + (layerNum+1) + '=================');
            // console.log(nextLayer);
            // console.log('===============================================');


            for (let n = 0; n < hiddenLayer.length-1; n++) {
                hiddenLayer[n].calcHiddenGradients(nextLayer);
            }

            // console.log(hiddenLayer);
        }
        //console.log(this.layers);

        // For all layers from ouput to first hidden layer
        // update connection weights

        for (let layerNum = this.layers.length - 1; layerNum > 0; layerNum--) {
            var layer = this.layers[layerNum];
            var prevLayer = this.layers[layerNum - 1];

            for (let n = 0; n < layer.length - 1; n++) {
                layer[n].updateInputWeights(prevLayer);
            }
        }

        // console.log(this.la);
    }
}

// simple example
// var net = new Net([2, 3, 1]);

// var epochCount = 1000;
// for (let index = 0; index < epochCount; index++) {
//     net.feedForward([1,0]);
//     net.backProp([1]);
//     net.feedForward([0,1]);
//     net.backProp([1]);
//     net.feedForward([0,0]);
//     net.backProp([0]);
//     net.feedForward([1,1]);
//     net.backProp([0]);
// }

// net.feedForward([1,1]);
//--------------------------------

//iris fisher example
var nn_iris = [[5.1,3.5,1.4,0.2,"setosa"],[4.9,3,1.4,0.2,"setosa"],[4.7,3.2,1.3,0.2,"setosa"],[4.6,3.1,1.5,0.2,"setosa"],[5,3.6,1.4,0.2,"setosa"],[5.4,3.9,1.7,0.4,"setosa"],[4.6,3.4,1.4,0.3,"setosa"],[5,3.4,1.5,0.2,"setosa"],[4.4,2.9,1.4,0.2,"setosa"],[4.9,3.1,1.5,0.1,"setosa"],[5.4,3.7,1.5,0.2,"setosa"],[4.8,3.4,1.6,0.2,"setosa"],[4.8,3,1.4,0.1,"setosa"],[4.3,3,1.1,0.1,"setosa"],[5.8,4,1.2,0.2,"setosa"],[5.7,4.4,1.5,0.4,"setosa"],[5.4,3.9,1.3,0.4,"setosa"],[5.1,3.5,1.4,0.3,"setosa"],[5.7,3.8,1.7,0.3,"setosa"],[5.1,3.8,1.5,0.3,"setosa"],[5.4,3.4,1.7,0.2,"setosa"],[5.1,3.7,1.5,0.4,"setosa"],[4.6,3.6,1,0.2,"setosa"],[5.1,3.3,1.7,0.5,"setosa"],[4.8,3.4,1.9,0.2,"setosa"],[5,3,1.6,0.2,"setosa"],[5,3.4,1.6,0.4,"setosa"],[5.2,3.5,1.5,0.2,"setosa"],[5.2,3.4,1.4,0.2,"setosa"],[4.7,3.2,1.6,0.2,"setosa"],[4.8,3.1,1.6,0.2,"setosa"],[5.4,3.4,1.5,0.4,"setosa"],[5.2,4.1,1.5,0.1,"setosa"],[5.5,4.2,1.4,0.2,"setosa"],[4.9,3.1,1.5,0.1,"setosa"],[5,3.2,1.2,0.2,"setosa"],[5.5,3.5,1.3,0.2,"setosa"],[4.9,3.1,1.5,0.1,"setosa"],[4.4,3,1.3,0.2,"setosa"],[5.1,3.4,1.5,0.2,"setosa"],[5,3.5,1.3,0.3,"setosa"],[4.5,2.3,1.3,0.3,"setosa"],[4.4,3.2,1.3,0.2,"setosa"],[5,3.5,1.6,0.6,"setosa"],[5.1,3.8,1.9,0.4,"setosa"],[4.8,3,1.4,0.3,"setosa"],[5.1,3.8,1.6,0.2,"setosa"],[4.6,3.2,1.4,0.2,"setosa"],[5.3,3.7,1.5,0.2,"setosa"],[5,3.3,1.4,0.2,"setosa"],[7,3.2,4.7,1.4,"versicolor"],[6.4,3.2,4.5,1.5,"versicolor"],[6.9,3.1,4.9,1.5,"versicolor"],[5.5,2.3,4,1.3,"versicolor"],[6.5,2.8,4.6,1.5,"versicolor"],[5.7,2.8,4.5,1.3,"versicolor"],[6.3,3.3,4.7,1.6,"versicolor"],[4.9,2.4,3.3,1,"versicolor"],[6.6,2.9,4.6,1.3,"versicolor"],[5.2,2.7,3.9,1.4,"versicolor"],[5,2,3.5,1,"versicolor"],[5.9,3,4.2,1.5,"versicolor"],[6,2.2,4,1,"versicolor"],[6.1,2.9,4.7,1.4,"versicolor"],[5.6,2.9,3.6,1.3,"versicolor"],[6.7,3.1,4.4,1.4,"versicolor"],[5.6,3,4.5,1.5,"versicolor"],[5.8,2.7,4.1,1,"versicolor"],[6.2,2.2,4.5,1.5,"versicolor"],[5.6,2.5,3.9,1.1,"versicolor"],[5.9,3.2,4.8,1.8,"versicolor"],[6.1,2.8,4,1.3,"versicolor"],[6.3,2.5,4.9,1.5,"versicolor"],[6.1,2.8,4.7,1.2,"versicolor"],[6.4,2.9,4.3,1.3,"versicolor"],[6.6,3,4.4,1.4,"versicolor"],[6.8,2.8,4.8,1.4,"versicolor"],[6.7,3,5,1.7,"versicolor"],[6,2.9,4.5,1.5,"versicolor"],[5.7,2.6,3.5,1,"versicolor"],[5.5,2.4,3.8,1.1,"versicolor"],[5.5,2.4,3.7,1,"versicolor"],[5.8,2.7,3.9,1.2,"versicolor"],[6,2.7,5.1,1.6,"versicolor"],[5.4,3,4.5,1.5,"versicolor"],[6,3.4,4.5,1.6,"versicolor"],[6.7,3.1,4.7,1.5,"versicolor"],[6.3,2.3,4.4,1.3,"versicolor"],[5.6,3,4.1,1.3,"versicolor"],[5.5,2.5,4,1.3,"versicolor"],[5.5,2.6,4.4,1.2,"versicolor"],[6.1,3,4.6,1.4,"versicolor"],[5.8,2.6,4,1.2,"versicolor"],[5,2.3,3.3,1,"versicolor"],[5.6,2.7,4.2,1.3,"versicolor"],[5.7,3,4.2,1.2,"versicolor"],[5.7,2.9,4.2,1.3,"versicolor"],[6.2,2.9,4.3,1.3,"versicolor"],[5.1,2.5,3,1.1,"versicolor"],[5.7,2.8,4.1,1.3,"versicolor"],[6.3,3.3,6,2.5,"virginica"],[5.8,2.7,5.1,1.9,"virginica"],[7.1,3,5.9,2.1,"virginica"],[6.3,2.9,5.6,1.8,"virginica"],[6.5,3,5.8,2.2,"virginica"],[7.6,3,6.6,2.1,"virginica"],[4.9,2.5,4.5,1.7,"virginica"],[7.3,2.9,6.3,1.8,"virginica"],[6.7,2.5,5.8,1.8,"virginica"],[7.2,3.6,6.1,2.5,"virginica"],[6.5,3.2,5.1,2,"virginica"],[6.4,2.7,5.3,1.9,"virginica"],[6.8,3,5.5,2.1,"virginica"],[5.7,2.5,5,2,"virginica"],[5.8,2.8,5.1,2.4,"virginica"],[6.4,3.2,5.3,2.3,"virginica"],[6.5,3,5.5,1.8,"virginica"],[7.7,3.8,6.7,2.2,"virginica"],[7.7,2.6,6.9,2.3,"virginica"],[6,2.2,5,1.5,"virginica"],[6.9,3.2,5.7,2.3,"virginica"],[5.6,2.8,4.9,2,"virginica"],[7.7,2.8,6.7,2,"virginica"],[6.3,2.7,4.9,1.8,"virginica"],[6.7,3.3,5.7,2.1,"virginica"],[7.2,3.2,6,1.8,"virginica"],[6.2,2.8,4.8,1.8,"virginica"],[6.1,3,4.9,1.8,"virginica"],[6.4,2.8,5.6,2.1,"virginica"],[7.2,3,5.8,1.6,"virginica"],[7.4,2.8,6.1,1.9,"virginica"],[7.9,3.8,6.4,2,"virginica"],[6.4,2.8,5.6,2.2,"virginica"],[6.3,2.8,5.1,1.5,"virginica"],[6.1,2.6,5.6,1.4,"virginica"],[7.7,3,6.1,2.3,"virginica"],[6.3,3.4,5.6,2.4,"virginica"],[6.4,3.1,5.5,1.8,"virginica"],[6,3,4.8,1.8,"virginica"],[6.9,3.1,5.4,2.1,"virginica"],[6.7,3.1,5.6,2.4,"virginica"],[6.9,3.1,5.1,2.3,"virginica"],[5.8,2.7,5.1,1.9,"virginica"],[6.8,3.2,5.9,2.3,"virginica"],[6.7,3.3,5.7,2.5,"virginica"],[6.7,3,5.2,2.3,"virginica"],[6.3,2.5,5,1.9,"virginica"],[6.5,3,5.2,2,"virginica"],[6.2,3.4,5.4,2.3,"virginica"],[5.9,3,5.1,1.8,"virginica"]];
var net = new Net([4, 8, 4, 1]);

var iris_category = {
    'setosa': [0],
    'versicolor': [0.5],
    'virginica': [1],
};

var epochCount = 65000;
var iteration = 0;
for (let index = 0; index < epochCount; index++) {
// while (net.error > 0.0035) {
    iteration++;
    nn_iris.forEach(iris => {
        //let learningSet = [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width];
        let learningSet = [iris[0], iris[1], iris[2], iris[3]];
        net.feedForward(learningSet);
        net.backProp(iris_category[iris[4]]);
    });
    // net.feedForward(learningSet);
    console.log(net.error, 'epoch num: ' + iteration)
}

//[5.5,2.6,4.4,1.2,"versicolor"]
console.log('Epoch: ' + iteration, 'Error = ' + net.error)
test_net([5.9,3.2,4.8,1.8,"versicolor"]);
test_net([5.5,2.6,4.4,1.2,"versicolor"]);
test_net([5,2.3,3.3,1,"versicolor"]);
test_net([7,3.2,4.7,1.4,"versicolor"]);
test_net([5.7,2.9,4.2,1.3,"versicolor"]);
test_net([6.8,3.2,5.9,2.3,"virginica"]);
test_net([6.7,3,5.2,2.3,"virginica"]);
test_net([5,3.5,1.3,0.3,"setosa"]);



function test_net(set) {
    var testNN = [set[0], set[1], set[2], set[3]];
    var rightAnswer = set[4];


    net.feedForward(testNN)
    var resultSet = net.getResults();
    var flowerName = '';
    if(resultSet[0] < 0.2) {
        flowerName = 'sectosa';
    } else if(resultSet[0] > 0.4 &&  resultSet[0] < 0.6) {
        flowerName = 'versicolor';
    } else if(resultSet[0] > 0.8) {
        flowerName = 'virginica';
    }

    console.log('result ' + resultSet[0], 'flower: ' + flowerName, 'right: ' + rightAnswer);
}

