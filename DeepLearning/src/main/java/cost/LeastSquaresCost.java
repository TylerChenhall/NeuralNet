package cost;

import tensor.TensorV0;

/**
 * @author tyler
 */
public class LeastSquaresCost implements Cost {

    @Override
    public double computeCost(TensorV0 prediction, TensorV0 groundTruth) {
        var m = prediction.ncols;
        
        var costTerms = prediction.subtract(groundTruth)
                .power(TensorV0.constant(2.0));
        var totalCost = costTerms.rowSum().columnSum();
        var averageCost = totalCost.multiply(TensorV0.constant(0.5 / m));
        return averageCost.value(0, 0);
    }

    @Override
    public TensorV0 computeCostDerivative(TensorV0 prediction, TensorV0 groundTruth) {
        // Drop the 1/m factor
        var derivatives = prediction.subtract(groundTruth);
        
        return derivatives;
    }
    
}
