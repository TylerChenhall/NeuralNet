package activation;

import tensor.TensorV0;

public class Activation {
    private final ActivationType type;
    
    public Activation(ActivationType type) {
        this.type = type;
    }
    
    public TensorV0 apply(TensorV0 input) {
        switch(type) {
            case Linear:
                return input;
            case ReLU:
                return input.relu();
            case Sigmoid:
                return input.sigmoid();
            case Softmax:
                TensorV0 exponentials = input.exponentiate();
                TensorV0 sums = exponentials.columnSum();
                return exponentials.divideBy(sums);
            case Tanh:
                return input.tanh();
            default:
                throw new UnsupportedOperationException("Type: " + type + " is not yet supported.");
        }
    }
    
    public TensorV0 derivateApply(TensorV0 dInput, TensorV0 cacheZ) {
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
