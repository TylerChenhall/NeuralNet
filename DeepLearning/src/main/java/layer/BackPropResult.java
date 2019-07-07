package layer;

import java.util.Map;
import tensor.TensorV0;

/**
 * Basic encapsulation object for the backpropagation of a single layer.
 * 
 * dA represents the values passed back to the previous layer
 * dParameters is a map containing computed derivative terms relevant for
 * updating the parameters of this layer.
 * 
 * @author tyler
 */
public class BackPropResult {
    public final TensorV0 dA;
    public final Map<String, TensorV0> dParameters;
    
    public BackPropResult(TensorV0 dA, Map<String, TensorV0> dParameters) {
        this.dA = dA;
        this.dParameters = dParameters;
    }
}
