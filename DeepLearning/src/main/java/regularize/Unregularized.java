package regularize;

import tensor.Tensor;
import tensor.Tensor2D;

/**
 * @author tyler
 */
public class Unregularized implements Regularizer {

    @Override
    public Tensor computeRegularizedDerivatives(Tensor parameters) {
        return Tensor2D.constant(0.0);
    }

    @Override
    public double getRegularizerCost() {
        return 0.0;
    }

    @Override
    public void resetRunningCost() {
        return;
    }
    
    @Override
    public void setBatchSize(int m) {
        return;
    }
    
}
