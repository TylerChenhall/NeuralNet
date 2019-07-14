package regularize;

import tensor.Tensor;
import tensor.Tensor2D;

/**
 * @author tyler
 */
public class L2Regularizer implements Regularizer {

    private final double learningRate;
    private final double lambda;
    private int m;

    private double regularizerCost = 0.0;

    public L2Regularizer(double learningRate, double lambda) {
        this.learningRate = learningRate;
        this.lambda = lambda;
        m = 1; // Default value
    }

    @Override
    public Tensor computeRegularizedDerivatives(Tensor parameters) {
        double factor = learningRate * lambda / m;

        double cost = parameters.multiply(parameters).allSum()
                .multiply(Tensor2D.constant(factor / 2.0)).value2(0, 0, 0, 0);
        regularizerCost += cost;

        return parameters.multiply(Tensor2D.constant(factor));
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
