package layer;

import java.util.Map;
import tensor.Tensor;

/**
 * Basic encapsulation object for the backward propagation of a single layer.
 * 
 * dA represents the values passed back to the previous layer
 * dParameters is a map containing computed derivative terms relevant for
 * updating the parameters of this layer.
 * 
 * @author tyler
 */
public class BackPropResult {
    public final Tensor dA;
    public final Map<String, Tensor> dParameters;
    
    public BackPropResult(Tensor dA, Map<String, Tensor> dParameters) {
        this.dA = dA;
        this.dParameters = dParameters;
    }
}
