package layer;

import java.util.Map;
import tensor.Tensor;

/**
 * Basic encapsulation object for the forward propagation of a single layer.
 * 
 * @author tyler
 */
public class ForwardPropResult {
    public final Tensor a;
    public final Map<String, Tensor> cache;
    
    public ForwardPropResult(Tensor a, Map<String, Tensor> cache) {
        this.a = a;
        this.cache = cache;
    }
}
