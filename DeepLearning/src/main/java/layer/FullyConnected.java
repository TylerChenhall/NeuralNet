package layer;

import activation.Activation;
import java.util.HashMap;
import java.util.Map;
import tensor.TensorV0;
import tensor.TensorV0Builder;

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
public class FullyConnected {
    public static final String OLD_ACTIVATION = "a_old";
    public static final String PRE_ACTIVATION = "z";
    public static final String POST_ACTIVATION = "a";
    public static final String D_WEIGHTS = "dW";
    public static final String D_BIAS = "db";
    
    private final Activation activation;
    
    // Weights for all nodes in this layer of the network. Each row represents
    // the weights for a single node.
    private TensorV0 weights;
    private TensorV0 bias;
    
    public FullyConnected(Activation activation, int nNodes, int inputDim) {
        this.activation = activation;
        
        // TODO: provide option to skip the bias here or in training
        // TODO: option to customize intialization
        weights = TensorV0Builder.heInitialization(nNodes, inputDim);
        bias = new TensorV0(nNodes, 1);
        
    }
    
    public FullyConnected(Activation activation, TensorV0 weights, TensorV0 bias) {
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
     * @return Returns a map containing the post and pre-activation outputs
     */
    public Map<String,TensorV0> forwardPropagate(TensorV0 x) {
        TensorV0 z = weights.matrixMultiply(x).add(bias);
        TensorV0 a = activation.apply(z);
        
        HashMap<String, TensorV0> results = new HashMap<>();
        results.put(OLD_ACTIVATION, x);
        results.put(PRE_ACTIVATION, z);
        results.put(POST_ACTIVATION, a);
        
        return results;
    }
    
    /**
     * Computes derivatives needed for backpropagation in previous layers.
     * 
     * Formulas:
     * dW = 1/m * dZ*A_prev^T
     * db = 1/m * dZ.rowSum
     * 
     * @param dA
     * @param cache Map containing the activation values, as output by forwardPropagate
     * @return 
     */
    public BackPropResult backwardPropagate(TensorV0 dA, Map<String, TensorV0> cache) {
        var z = cache.get(PRE_ACTIVATION);
        var aOld = cache.get(OLD_ACTIVATION);
        var factor = TensorV0.constant(1.0 / aOld.ncols);
        
        var dZ = activation.derivateApply(dA, z);
        var dW = dZ.matrixMultiply(aOld.transpose()).multiply(factor);
        var db = dZ.rowSum().multiply(factor);
        
        TensorV0 daPrev = weights.transpose().matrixMultiply(dZ);
        HashMap<String, TensorV0> results = new HashMap<>();
        results.put(D_WEIGHTS, dW);
        results.put(D_BIAS, db);
        
        return new BackPropResult(daPrev, results);
    }
}
