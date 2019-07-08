package network;

import cost.CrossEntropyCost;
import java.util.ArrayList;
import java.util.List;
import layer.BackPropResult;
import layer.ForwardPropResult;
import layer.FullyConnected;
import optimize.GradientDescent;
import tensor.TensorV0;

public class NeuralNetwork {

    private final CrossEntropyCost costFunction;
    private final List<FullyConnected> layers;
    private final GradientDescent optimizer;

    public NeuralNetwork(List<FullyConnected> layers, CrossEntropyCost costFunction, GradientDescent optimizer) {
        this.layers = layers;
        this.costFunction = costFunction;
        this.optimizer = optimizer;
    }

    public ArrayList<Double> train(TensorV0 dataFeatures, TensorV0 dataLabels, int epochs) {
        var epochCosts = new ArrayList<Double>();
        for (int i = 0; i < epochs; i++) {
            // TODO: at some point, implement mini-batch support
            
            // Forward propagation
            var fp = new ArrayList<ForwardPropResult>();
            var activation = dataFeatures;
            for (FullyConnected layer : layers) {
                var layerResult = layer.forwardPropagate(activation);
                activation = layerResult.a;
                fp.add(layerResult);
            }
            
            // Cost
            var cost = costFunction.computeCost(activation, dataLabels);
            epochCosts.add(cost);
            
            var dA = costFunction.computeCostDerivative(activation, dataLabels);
            
            // Backward propagation
            var bp = new BackPropResult[layers.size()];// new ArrayList<BackPropResult>(layers.size());
            for (int j = layers.size() - 1; j >= 0; j--) {
                var layer = layers.get(j);
                var layerResult = layer.backwardPropagate(dA, fp.get(j));
                //bp.set(j, layerResult);
                bp[j] = layerResult;
                dA = layerResult.dA;
            }
            
            // Parameter updates
            for (int j = 0; j < layers.size(); j++) {
                var dParameters = bp[j].dParameters; //bp.get(j).dParameters;
                var deltaParameters = optimizer.computeParameterUpdates(dParameters, j);
                layers.get(j).updateParameters(deltaParameters);
            }
        }
        return epochCosts;
    }

    /**
     * Applies the neural network model to the given data
     *
     * @param dataFeatures TensorV0 of data points (1 point per column)
     * @return TensorV0 of predictions (1 prediction per column)
     */
    public TensorV0 predict(TensorV0 dataFeatures) {
        var activation = dataFeatures;
        for (FullyConnected layer : layers) {
            var layerResult = layer.forwardPropagate(activation);
            activation = layerResult.a;
        }
        return activation;
    }

    public double evaluate(TensorV0 dataFeatures, TensorV0 dataLabels) {
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
