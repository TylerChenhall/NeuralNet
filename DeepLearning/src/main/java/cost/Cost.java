package cost;

import tensor.Tensor;
import tensor.TensorV0;

/**
 * @author tyler
 */
public interface Cost {
    public double computeCost(Tensor prediction, Tensor groundTruth);
    
    public Tensor computeCostDerivative(Tensor prediction, Tensor groundTruth);
}
