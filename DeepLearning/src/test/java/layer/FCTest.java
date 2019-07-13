package layer;

import activation.Activation;
import activation.ActivationType;
import cost.CrossEntropyCost;
import java.util.ArrayList;
import network.NeuralNetwork;
import optimize.Adam;
import optimize.GradientDescent;
import optimize.Momentum;
import optimize.RMSProp;
import tensor.Tensor2D;

public class FCTest {
    public static void main(String[] args) {
        // TODO: eventually convert this to more of a standard test
        
        // Create a single unit sigmoid layer. It models the classification
        // boundary 1.0 * x0 + 2.0 * x1 - 3.0 >= 0.0
        double[][] weights = {{1.0, 2.0}};
        var layer = new FullyConnected(new Activation(ActivationType.Sigmoid),
                new Tensor2D(weights), Tensor2D.constant(-3.0));
        
        // Create a quick dataset
        double[][] inputs = {
            {-5.0, -5.0, -5.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0},
            {-5.0, 0.0, 5.0, -5.0, 0.0, 5.0, -5.0, 0.0, 5.0}};
        var inputTensor = new Tensor2D(inputs);
        
        double[][] classes = {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0}};
        var groundTruths = new Tensor2D(classes);
        
        // Propagate the layer on the dataset
        var results = layer.forwardPropagate(inputTensor, false);
        var predictions = results.a;
        
        // Evaluate results
        // Compute cost
        var costFunction = new CrossEntropyCost();
        var cost = costFunction.computeCost(predictions, groundTruths);
        var costDerivatives = costFunction.computeCostDerivative(predictions, groundTruths);
        
        System.out.println("Raw layer results");
        System.out.println("-----------------");
        System.out.println(groundTruths);
        System.out.println(predictions);
        System.out.println(cost);
        //System.out.println(costDerivatives);
        
        var list = new ArrayList<Layer>();
        list.add(layer);
        var optimizer = new GradientDescent(0.12);
        
        var network = new NeuralNetwork(list, costFunction, optimizer);
        
        System.out.println("Network results");
        System.out.println("-----------------");
        System.out.println(groundTruths);
        System.out.println(network.predict(inputTensor));
        System.out.println(network.evaluate(inputTensor, groundTruths));
        
        // Training a network
        System.out.println("Trained network results");
        System.out.println("-----------------");
        
        var untrainedLayer = new FullyConnected(
                new Activation(ActivationType.Sigmoid), 1, 2);
        var untrainedList = new ArrayList<Layer>();
        untrainedList.add(untrainedLayer);
        
        // Confirmed to work:
        //   GradientDescent(0.12)
        //   Momentum(0.12, 0.9)
        //   RMSProp(0.12, 0.999)
        //   Adam(0.06, 0.9, 0.999) - lower learning rate required for convergence
        var trainingOptimizer = new Adam(0.06, 0.9, 0.999);
        var trainableNetwork = 
                new NeuralNetwork(untrainedList, costFunction, trainingOptimizer);
        
        var initialCost = trainableNetwork.evaluate(inputTensor, groundTruths);
        
        int numEpochs = 100;
        var epochCosts = trainableNetwork.train(inputTensor, groundTruths, numEpochs);
        
        System.out.println(epochCosts.get(0));
        System.out.println(epochCosts.get(numEpochs-1));
        System.out.println(groundTruths);
        System.out.println(trainableNetwork.predict(inputTensor));
        System.out.println(trainableNetwork.evaluate(inputTensor, groundTruths));
        
        System.out.println(trainableNetwork);
    }
}
