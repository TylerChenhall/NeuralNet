package network;

import cost.Cost;
import java.util.ArrayList;
import java.util.List;
import layer.BackPropResult;
import layer.ForwardPropResult;
import layer.Layer;
import optimize.Optimizer;
import regularize.Regularizer;
import regularize.Unregularized;
import tensor.Tensor;

public class NeuralNetwork {

    private final Cost costFunction;
    private final List<Layer> layers;
    private final Optimizer optimizer;
    private final Regularizer regularizer;
    
    public NeuralNetwork(List<Layer> layers, Cost costFunction, Optimizer optimizer, Regularizer regularizer) {
        this.layers = layers;
        this.costFunction = costFunction;
        this.optimizer = optimizer;
        this.regularizer = regularizer;
    }

    public NeuralNetwork(List<Layer> layers, Cost costFunction, Optimizer optimizer) {
        this.layers = layers;
        this.costFunction = costFunction;
        this.optimizer = optimizer;
        this.regularizer = new Unregularized();
    }

    public ArrayList<Double> train(Tensor dataFeatures, Tensor dataLabels, int epochs) {
        var epochCosts = new ArrayList<Double>();
        for (int i = 0; i < epochs; i++) {
            // For now, m is fixed, but this may vary when using mini-batches.
            // TODO: at some point, implement mini-batch support
            regularizer.setBatchSize(dataFeatures.mDim());
            
            // Forward propagation
            var fp = new ArrayList<ForwardPropResult>();
            var activation = dataFeatures;
            for (var layer : layers) {
                var layerResult = layer.forwardPropagate(activation, true);
                activation = layerResult.a;
                fp.add(layerResult);
            }
            
            var dA = costFunction.computeCostDerivative(activation, dataLabels);
            
            // Backward propagation
            var bp = new BackPropResult[layers.size()];
            for (int j = layers.size() - 1; j >= 0; j--) {
                var layer = layers.get(j);
                var layerResult = layer.backwardPropagate(dA, fp.get(j));
                bp[j] = layerResult;
                dA = layerResult.dA;
            }
            
            // Parameter updates
            for (int j = 0; j < layers.size(); j++) {
                var dParameters = bp[j].dParameters;
                var deltaParameters = optimizer.computeParameterUpdates(dParameters, j);
                layers.get(j).updateParameters(deltaParameters, regularizer);
            }
            
            // Cost
            var cost = costFunction.computeCost(activation, dataLabels)
                    + regularizer.getRegularizerCost();
            epochCosts.add(cost);
            regularizer.resetRunningCost();
        }
        return epochCosts;
    }

    /**
     * Applies the neural network model to the given data
     *
     * @param dataFeatures Tensor of data points (1 point per row)
     * @return Tensor of predictions (1 prediction per row)
     */
    public Tensor predict(Tensor dataFeatures) {
        var activation = dataFeatures;
        for (var layer : layers) {
            var layerResult = layer.forwardPropagate(activation, false);
            activation = layerResult.a;
        }
        return activation;
    }

    public double evaluate(Tensor dataFeatures, Tensor dataLabels) {
        var predictions = predict(dataFeatures);
        return costFunction.computeCost(predictions, dataLabels);
    }
    
    public String toString() {
        var sb = new StringBuilder();
        for (int i = 0; i < layers.size(); i++) {
            var layer = layers.get(i);
            sb.append("Layer ");
            sb.append(i);
            sb.append(": ");
            sb.append(System.lineSeparator());
            sb.append(layer);
            if (i != layers.size()-1) {
                sb.append(System.lineSeparator().repeat(2));
            }
        }
        
        return sb.toString();
    }
}
