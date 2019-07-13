package layer;

import java.util.Map;
import tensor.Tensor;

/**
 * @author tyler
 */
public interface Layer {
    public ForwardPropResult forwardPropagate(Tensor x, boolean training);
    
    public BackPropResult backwardPropagate(Tensor dA, ForwardPropResult cache);
    
    public void updateParameters(Map<String, Tensor> deltaParameters);
}
