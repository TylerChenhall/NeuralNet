package layer;

import activation.Activation;
import java.util.HashMap;
import java.util.Map;
import regularize.Regularizer;
import tensor.Tensor;
import tensor.Tensor2D;
import tensor.TensorBuilder;

/**
 * Basic implementation of a Fully Connected Neural Network layer.
 * 
 * This is still in progress, and I'm trying to decide what the API will be.
 * The API may evolve if I try to create a common interface with RNN, CNN layers
 * 
 * TODO Initial inputs
 *  Initialization function
 *  Whether to include a bias unit (not needed if doing batch normalization)
 *  
 * @author tyler
 */
public class FullyConnected implements Layer {
    public static final String OLD_ACTIVATION = "a_old";
    public static final String PRE_ACTIVATION = "z";
    public static final String POST_ACTIVATION = "a";
    public static final String D_WEIGHTS = "dW";
    public static final String D_BIAS = "db";
    
    private final Activation activation;
    
    // Weights for all nodes in this layer of the network. Each column
    // represents the weights for a single node.
    private Tensor2D weights;
    private Tensor2D bias;
    
    public FullyConnected(Activation activation, int nNodes, int inputDim) {
        this.activation = activation;
        
        // TODO: provide option to skip the bias here or in training
        // TODO: option to customize intialization
        weights = TensorBuilder.heInitialization(inputDim, nNodes);
        bias = new Tensor2D(1, nNodes);
        
    }
    
    public FullyConnected(Activation activation, Tensor2D weights, Tensor2D bias) {
        this.activation = activation;
        this.weights = weights;
        this.bias = bias;
    }
    
    /**
     * Computes the activations from a fully connected layer based on the input.
     * 
     * We assume the results for each data point are organized into rows of
     * x. This is also true of the output.
     * 
     * @param x Vectorized inputs 
     * @param training Whether the network is currently being trained
     * @return Returns a map containing the post and pre-activation outputs
     */
    @Override
    public ForwardPropResult forwardPropagate(Tensor x, boolean training) {
        if (!(x instanceof Tensor2D)) {
            throw new IllegalArgumentException("Input for fully connected layers must be 2D Tensors.");
        }
        var aOld = (Tensor2D) x;
        var z = aOld.matrixMultiply(weights).add(bias);
        var a = activation.apply(z);
        
        var cache = new HashMap<String, Tensor>();
        cache.put(OLD_ACTIVATION, aOld);
        cache.put(PRE_ACTIVATION, z);
        
        return new ForwardPropResult(a, cache);
    }
    
    /**
     * Computes derivatives needed for backward propagation in previous layers.
     * 
     * Formulas:
     * dW = 1/m * A_prev^T*dZ
     * db = 1/m * dZ.columnSum
     * 
     * @param dA
     * @param cache Map containing the activation values, as output by forwardPropagate
     * @return 
     */
    @Override
    public BackPropResult backwardPropagate(Tensor dA, ForwardPropResult cache) {
        var z = cache.cache.get(PRE_ACTIVATION);
        var aOld = (Tensor2D) cache.cache.get(OLD_ACTIVATION);
        var factor = Tensor2D.constant(1.0 / aOld.mDim());
        
        var dZ = (Tensor2D) activation.derivateApply(dA, z);
        var dW = aOld.transpose().matrixMultiply(dZ).multiply(factor);
        var db = dZ.columnSum().multiply(factor);
        
        var daPrev = dZ.matrixMultiply(weights.transpose());
        HashMap<String, Tensor> results = new HashMap<>();
        results.put(D_WEIGHTS, dW);
        results.put(D_BIAS, db);
        
        return new BackPropResult(daPrev, results);
    }
    
    @Override
    public void updateParameters(Map<String, Tensor> deltaParameters, Regularizer r) {
        weights = (Tensor2D) weights.add(deltaParameters.get(D_WEIGHTS))
                .subtract(r.computeRegularizedDerivatives(weights));
        bias = (Tensor2D) bias.add(deltaParameters.get(D_BIAS));
    }
    
    @Override
    public String toString() {
        var sb = new StringBuilder();
        sb.append("Fully Connected Layer");
        sb.append(System.lineSeparator());
        sb.append("Weights:");
        sb.append(System.lineSeparator());
        // To make toString more readable, we transpose the weights and bias
        // Tensors so each row gives the parameters for a single unit.
        sb.append(weights.transpose());
        sb.append(System.lineSeparator());
        sb.append("Bias:");
        sb.append(System.lineSeparator());
        sb.append(bias.transpose());
        
        return sb.toString();
    }
}
