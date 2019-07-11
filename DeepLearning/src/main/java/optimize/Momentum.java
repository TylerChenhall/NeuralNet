package optimize;

import java.util.HashMap;
import java.util.Map;
import tensor.Tensor;
import tensor.TensorV0;

/**
 * @author tyler
 */
public class Momentum implements Optimizer {
    private final double learningRate;
    private final TensorV0 beta;
    private final Map<String, Tensor> momentums;
    
    public Momentum(double learningRate, double beta) {
        this.learningRate = learningRate;
        if (beta < 0 || beta > 1) {
            throw new IllegalArgumentException("Momentum beta parameter must be in [0,1].");
        }
        this.beta = TensorV0.constant(beta);
        momentums = new HashMap<>();
    }
    
    @Override
    public Map<String, Tensor> computeParameterUpdates(Map<String, Tensor> dParameters, int identifier) {
        var parameterUpdates = new HashMap<String, Tensor>();
        var factor = TensorV0.constant(-1.0 * learningRate);
        
        for (var key : dParameters.keySet()) {
            var lookup = key + identifier;
            var momentum = momentums.getOrDefault(lookup, TensorV0.constant(0.0));
            momentum = beta.multiply(momentum).add(
                    TensorV0.one().subtract(beta)
                            .multiply(dParameters.get(key)));
            
            momentums.put(lookup, momentum);
            parameterUpdates.put(key, momentum.multiply(factor));
        }
        return parameterUpdates;
    }
}
