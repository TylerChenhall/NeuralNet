package optimize;

import java.util.HashMap;
import java.util.Map;
import tensor.TensorV0;

/**
 * Implements ADAM (adaptive moment) optimization.
 * 
 * This essentially combines RMSprop and Momentum features.
 * 
 * Observation: Adam and RMSProp give bad results if run for too many epochs.
 * The cost seems to minimize then blow up. This may be due to variances being 
 * too close to zero after approximate convergence occurs.
 * 
 * @author tyler
 */
public class Adam implements Optimizer {
    private final double learningRate;
    private final TensorV0 beta1;
    private final TensorV0 beta2;
    private final TensorV0 epsilon = TensorV0.constant(1.0e-8);
    private final Map<String, TensorV0> momentums;
    private final Map<String, TensorV0> variances;
    
    /**
     * Construct an ADAM optimizer.
     * 
     * Reasonable default values are beta1 ~ 0.9, beta2 ~ 0.999
     * 
     * @param learningRate
     * @param beta1 Parameter for exponentially weighted average of momentum
     * @param beta2 Parameter for exponentially weighted average of variance
     */
    public Adam(double learningRate, double beta1, double beta2) {
        this.learningRate = learningRate;
        if (beta1 < 0 || beta1 > 1 || beta2 < 0 || beta2 > 1) {
            throw new IllegalArgumentException("Momentum beta parameters must be in [0,1].");
        }
        this.beta1 = TensorV0.constant(beta1);
        this.beta2 = TensorV0.constant(beta2);
        momentums = new HashMap<>();
        variances = new HashMap<>();
    }
    
    @Override
    public Map<String, TensorV0> computeParameterUpdates(Map<String, TensorV0> dParameters, int identifier) {
        var parameterUpdates = new HashMap<String, TensorV0>();
        var factor = TensorV0.constant(-1.0 * learningRate);
        
        for (var key : dParameters.keySet()) {
            var lookup = key + identifier;
            var dParameter = dParameters.get(key);
            
            // Compute updated momentum
            var momentum = momentums.getOrDefault(lookup, TensorV0.constant(0.0));
            momentum = beta1.multiply(momentum).add(
                    TensorV0.one().subtract(beta1)
                            .multiply(dParameter));
            
            momentums.put(lookup, momentum);
            
            // Compute updated variance
            var variance = variances.getOrDefault(lookup, TensorV0.constant(0.0));
            variance = beta2.multiply(variance).add(
                    TensorV0.one().subtract(beta2)
                            .multiply(dParameter.multiply(dParameter)));
            
            variances.put(lookup, variance);
            
            var sd = variance.power(TensorV0.constant(0.5));
            
            // Divide by sd + epsilon to avoid division by zero.
            parameterUpdates.put(key, momentum
                    .multiply(factor)
                    .divideBy(sd.add(epsilon)));
        }
        return parameterUpdates;
    }
}
