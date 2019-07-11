package cost;

import tensor.Tensor;
import tensor.Tensor2D;

/**
 * @author tyler
 */
public class LeastSquaresCost implements Cost {

    @Override
    public double computeCost(Tensor prediction, Tensor groundTruth) {
        var m = prediction.mDim();
        
        var costTerms = prediction.subtract(groundTruth)
                .power(Tensor2D.constant(2.0));
        var totalCost = costTerms.allSum();
        var averageCost = totalCost.multiply(Tensor2D.constant(0.5 / m));
        return averageCost.value(0, 0);
    }

    @Override
    public Tensor computeCostDerivative(Tensor prediction, Tensor groundTruth) {
        // Drop the 1/m factor
        var derivatives = prediction.subtract(groundTruth);
        
        return derivatives;
    }
    
}
