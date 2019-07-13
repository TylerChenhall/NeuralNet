package layer;

import activation.Activation;
import java.util.HashMap;
import java.util.Map;
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
    
    // Weights for all nodes in this layer of the network. Each row represents
    // the weights for a single node.
    private Tensor2D weights;
    private Tensor2D bias;
    
    public FullyConnected(Activation activation, int nNodes, int inputDim) {
        this.activation = activation;
        
        // TODO: provide option to skip the bias here or in training
        // TODO: option to customize intialization
        weights = TensorBuilder.heInitialization(nNodes, inputDim);
        bias = new Tensor2D(nNodes, 1);
        
    }
    
    public FullyConnected(Activation activation, Tensor2D weights, Tensor2D bias) {
        this.activation = activation;
        this.weights = weights;
        this.bias = bias;
    }
    
    /**
     * Computes the activations from a fully connected layer based on the input.
     * 
     * We assume the results for each data point are organized into columns of
     * x. This is also true of the output.
     * 
     * @param x Vectorized inputs 
     * @param training Whether the network is currently being trained
     * @return Returns a map containing the post and pre-activation outputs
     */
    public ForwardPropResult forwardPropagate(Tensor x, boolean training) {
        if (!(x instanceof Tensor2D)) {
            throw new IllegalArgumentException("Input for fully connected layers must be 2D Tensors.");
        }
        var aOld = (Tensor2D) x;
        var z = weights.matrixMultiply(aOld).add(bias);
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
     * dW = 1/m * dZ*A_prev^T
     * db = 1/m * dZ.rowSum
     * 
     * @param dA
     * @param cache Map containing the activation values, as output by forwardPropagate
     * @return 
     */
    public BackPropResult backwardPropagate(Tensor dA, ForwardPropResult cache) {
        var z = cache.cache.get(PRE_ACTIVATION);
        var aOld = (Tensor2D) cache.cache.get(OLD_ACTIVATION);
        var factor = Tensor2D.constant(1.0 / aOld.ncols);
        
        var dZ = (Tensor2D) activation.derivateApply(dA, z);
        var dW = dZ.matrixMultiply(aOld.transpose()).multiply(factor);
        var db = dZ.rowSum().multiply(factor);
        
        var daPrev = weights.transpose().matrixMultiply(dZ);
        HashMap<String, Tensor> results = new HashMap<>();
        results.put(D_WEIGHTS, dW);
        results.put(D_BIAS, db);
        
        return new BackPropResult(daPrev, results);
    }
    
    public void updateParameters(Map<String, Tensor> deltaParameters) {
        weights = (Tensor2D) weights.add(deltaParameters.get(D_WEIGHTS));
        bias = (Tensor2D) bias.add(deltaParameters.get(D_BIAS));
    }
    
    public String toString() {
        var sb = new StringBuilder();
        sb.append("Fully Connected Layer");
        sb.append(System.lineSeparator());
        sb.append("Weights:");
        sb.append(System.lineSeparator());
        sb.append(weights);
        sb.append(System.lineSeparator());
        sb.append("Bias:");
        sb.append(System.lineSeparator());
        sb.append(bias);
        
        return sb.toString();
    }
}
