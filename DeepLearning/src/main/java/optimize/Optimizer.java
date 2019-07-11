package optimize;

import java.util.Map;
import tensor.Tensor;

/**
 * @author tyler
 */
public interface Optimizer {
    public Map<String, Tensor> computeParameterUpdates(Map<String, Tensor> dParameters, int identifier);
}
