package optimize;

import java.util.Map;
import tensor.TensorV0;

/**
 * @author tyler
 */
public interface Optimizer {
    public Map<String, TensorV0> computeParameterUpdates(Map<String, TensorV0> dParameters, int identifier);
}
