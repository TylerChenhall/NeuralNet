package cost;

import tensor.TensorV0;

/**
 * @author tyler
 */
public interface Cost {
    public double computeCost(TensorV0 prediction, TensorV0 groundTruth);
    
    public TensorV0 computeCostDerivative(TensorV0 prediction, TensorV0 groundTruth);
}
