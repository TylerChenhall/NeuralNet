package optimize;

import java.util.HashMap;
import java.util.Map;
import tensor.TensorV0;

/**
 * @author tyler
 */
public class GradientDescent {
    final double learningRate;
    
    public GradientDescent(double learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * Computes the change in parameters to apply to be applied to the layer.
     * 
     * @param dParameters Map of derivatives (dCost / dparams)
     * @param layerNumber A unique index referring to the layer
     * @return Map of parameter updates, using the same keys as the input
     */
    public Map<String, TensorV0> computeParameterUpdates(Map<String, TensorV0> dParameters, int layerNumber) {
        var parameterUpdates = new HashMap<String, TensorV0>();
        var factor = TensorV0.constant(-1.0 * learningRate);
        
        for (var key : dParameters.keySet()) {
            parameterUpdates.put(key, dParameters.get(key).multiply(factor));
        }
        
        return parameterUpdates;
    }
}