package optimize;

import java.util.HashMap;
import java.util.Map;
import tensor.Tensor;
import tensor.Tensor2D;

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
    private final Tensor2D beta1;
    private final Tensor2D beta2;
    private final Tensor2D epsilon = Tensor2D.constant(1.0e-8);
    private final Map<String, Tensor> momentums;
    private final Map<String, Tensor> variances;
    
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
        this.beta1 = Tensor2D.constant(beta1);
        this.beta2 = Tensor2D.constant(beta2);
        momentums = new HashMap<>();
        variances = new HashMap<>();
    }
    
    @Override
    public Map<String, Tensor> computeParameterUpdates(Map<String, Tensor> dParameters, int identifier) {
        var parameterUpdates = new HashMap<String, Tensor>();
        var factor = Tensor2D.constant(-1.0 * learningRate);
        
        for (var key : dParameters.keySet()) {
            var lookup = key + identifier;
            var dParameter = dParameters.get(key);
            
            // Compute updated momentum
            var momentum = momentums.getOrDefault(lookup, Tensor2D.constant(0.0));
            momentum = beta1.multiply(momentum).add(Tensor2D.one().subtract(beta1)
                            .multiply(dParameter));
            
            momentums.put(lookup, momentum);
            
            // Compute updated variance
            var variance = variances.getOrDefault(lookup, Tensor2D.constant(0.0));
            variance = beta2.multiply(variance).add(Tensor2D.one().subtract(beta2)
                            .multiply(dParameter.multiply(dParameter)));
            
            variances.put(lookup, variance);
            
            var sd = variance.power(Tensor2D.constant(0.5));
            
            // Divide by sd + epsilon to avoid division by zero.
            parameterUpdates.put(key, momentum
                    .multiply(factor)
                    .divideBy(sd.add(epsilon)));
        }
        return parameterUpdates;
    }
}
