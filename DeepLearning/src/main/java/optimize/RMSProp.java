package optimize;

import java.util.HashMap;
import java.util.Map;
import tensor.Tensor;
import tensor.TensorV0;

/**
 * Implements the RMS Prop algorithm
 * 
 * @author tyler
 */
public class RMSProp implements Optimizer {
    private final double learningRate;
    private final TensorV0 beta;
    private final TensorV0 epsilon = TensorV0.constant(1.0e-8);
    private final Map<String, Tensor> variances;
    
    public RMSProp(double learningRate, double beta) {
        this.learningRate = learningRate;
        if (beta < 0 || beta > 1) {
            throw new IllegalArgumentException("Momentum beta parameter must be in [0,1].");
        }
        this.beta = TensorV0.constant(beta);
        variances = new HashMap<>();
    }
    
    @Override
    public Map<String, Tensor> computeParameterUpdates(Map<String, Tensor> dParameters, int identifier) {
        var parameterUpdates = new HashMap<String, Tensor>();
        var factor = TensorV0.constant(-1.0 * learningRate);
        
        for (var key : dParameters.keySet()) {
            var lookup = key + identifier;
            var variance = variances.getOrDefault(lookup, TensorV0.constant(0.0));
            var dParameter = dParameters.get(key);
            
            // Compute updated variance
            variance = beta.multiply(variance).add(
                    TensorV0.one().subtract(beta)
                            .multiply(dParameter.multiply(dParameter)));
            
            variances.put(lookup, variance);
            var sd = variance.power(TensorV0.constant(0.5));
            
            // Divide by sd + epsilon to avoid division by zero.
            parameterUpdates.put(key, dParameter
                    .multiply(factor)
                    .divideBy(sd.add(epsilon)));
        }
        return parameterUpdates;
    }
}
