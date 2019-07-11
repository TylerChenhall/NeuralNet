package activation;

import tensor.Tensor;
import tensor.TensorV0;

public class Activation {
    private final ActivationType type;
    
    public Activation(ActivationType type) {
        this.type = type;
    }
    
    public Tensor apply(Tensor input) {
        switch(type) {
            case Linear:
                return input;
            case ReLU:
                return input.relu();
            case Sigmoid:
                return input.sigmoid();
            case Softmax:
                if (!(input instanceof TensorV0)) {
                    throw new IllegalArgumentException("Softmax only supports TensorV0");
                }
                var exponentials = input.exponentiate();
                TensorV0 sums = ((TensorV0) exponentials).columnSum();
                return exponentials.divideBy(sums);
            case Tanh:
                return input.tanh();
            default:
                throw new UnsupportedOperationException("Type: " + type + " is not yet supported.");
        }
    }
    
    public Tensor derivateApply(Tensor dInput, Tensor cacheZ) {
        switch(type) {
            case Linear:
                return dInput;
            case ReLU:
                // Compute dA * (cacheZ >= 0)
                var mask = cacheZ.atLeast(TensorV0.constant(0.0));
                return dInput.multiply(mask);
            case Sigmoid:
                // Compute dA * [sigma * (1 - sigma)] with element-wise *
                var temp = cacheZ.sigmoid();
                var sigmoidDerivative = temp
                        .multiply(TensorV0.one().subtract(temp));
                return dInput.multiply(sigmoidDerivative);
            case Softmax:
                throw new UnsupportedOperationException("Type: " + type + " is not yet supported.");
            case Tanh:
                throw new UnsupportedOperationException("Type: " + type + " is not yet supported.");
            default:
                throw new UnsupportedOperationException("Type: " + type + " is not yet supported.");
        }
    }
}
