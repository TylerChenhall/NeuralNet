package cost;

import tensor.Tensor;
import tensor.Tensor2D;

/**
 * Computes cross entropy cost and cost derivative
 * @author tyler
 */
public class CrossEntropyCost implements Cost {
    
    /**
     * Computes the cross entropy cost function for a set of data points.
     * 
     * @param prediction Prediction tensor - each column is a single example
     * @param groundTruth Ground truth tensor - each column is a single example
     * @return Cost
     */
    @Override
    public double computeCost(Tensor prediction, Tensor groundTruth) {
        var m = prediction.mDim();
        
        var oneMinusPrediction = Tensor2D.one().subtract(prediction);
        var oneMinusTruth = Tensor2D.one().subtract(groundTruth);
        
        var costTerms = groundTruth.multiply(prediction.log())
                .add(oneMinusTruth.multiply(oneMinusPrediction.log()));
        var totalCost = costTerms.allSum();
        var averageCost = totalCost.multiply(Tensor2D.constant(-1.0 / m));
        return averageCost.value(0, 0);
    }
    
    /**
     * Computes element-wise derivatives of cost with respect to predictions.
     * 
     * The constant factor, 1/m is dropped
     * 
     * @param prediction Prediction tensor - each column is a single example
     * @param groundTruth Ground truth tensor - each column is a single example
     * @return Tensor of derivative terms
     */
    @Override
    public Tensor computeCostDerivative(Tensor prediction, Tensor groundTruth) {
        var oneMinusPrediction = Tensor2D.one().subtract(prediction);
        var oneMinusTruth = Tensor2D.one().subtract(groundTruth);
        
        var derivatives = oneMinusTruth.divideBy(oneMinusPrediction)
                .subtract(groundTruth.divideBy(prediction));
        
        return derivatives;
    }
}
