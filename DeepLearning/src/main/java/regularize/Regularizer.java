package regularize;

import tensor.Tensor;

/**
 * @author tyler
 */
public interface Regularizer {
    
    public Tensor computeRegularizedDerivatives(Tensor parameters);
    
    public double getRegularizerCost();
    
    public void resetRunningCost();
    
    public void setBatchSize(int m);
}
