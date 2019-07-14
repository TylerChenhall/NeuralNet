package regularize;

import tensor.Tensor;
import tensor.Tensor2D;

/**
 * @author tyler
 */
public class L1Regularizer implements Regularizer {

    private final double learningRate;
    private final double lambda;
    private int m;

    private double regularizerCost = 0.0;

    public L1Regularizer(double learningRate, double lambda) {
        this.learningRate = learningRate;
        this.lambda = lambda;
        this.m = 1; // Default value
    }

    @Override
    public Tensor computeRegularizedDerivatives(Tensor parameters) {
        var factor = Tensor2D.constant(learningRate * lambda / (2.0 * m));

        double cost = parameters.abs().allSum()
                .multiply(factor).value2(0, 0, 0, 0);
        regularizerCost += cost;

        return parameters.applyUnary(d -> (d > 0) ? 1.0 : ((d < 0) ? -1.0 : 0.0)
        ).multiply(factor);
    }

    @Override
    public double getRegularizerCost() {
        return regularizerCost;
    }

    @Override
    public void resetRunningCost() {
        regularizerCost = 0.0;
    }

    @Override
    public void setBatchSize(int m) {
        this.m = m;
    }

}
