package layer;

import java.util.Map;
import tensor.TensorV0;

/**
 * Basic encapsulation object for the forward propagation of a single layer.
 * 
 * @author tyler
 */
public class ForwardPropResult {
    public final TensorV0 a;
    public final Map<String, TensorV0> cache;
    
    public ForwardPropResult(TensorV0 a, Map<String, TensorV0> cache) {
        this.a = a;
        this.cache = cache;
    }
}
