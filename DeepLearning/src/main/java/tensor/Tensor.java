package tensor;

import java.text.DecimalFormat;
import java.util.function.BiFunction;
import java.util.function.Function;

/** 
 * A basic 2D tensor implementation.
 * 
 * TODO: consider adding a few more operations, such as
 *   concatenation or subsetting
 *   negation
 *   trig functions
 *   get / set operations
 *   
 * @author tyler
 */
public abstract class Tensor {
    public Tensor add(Tensor t) {
        return applyBinary(t, (d1, d2) -> d1 + d2);
    }

    public Tensor subtract(Tensor t) {
        return applyBinary(t, (d1, d2) -> d1 - d2);
    }

    public Tensor multiply(Tensor t) {
        return applyBinary(t, (d1, d2) -> d1 * d2);
    }

    public Tensor divideBy(Tensor t) {
        return applyBinary(t, (d1, d2) -> d1 / d2);
    }
    
    /**
     * Implements element-wise >= operator.
     * @param t
     * @return  The result is an indicator TensorV0, with 1.0 indicating true.
     */
    public Tensor atLeast(Tensor t) {
        return applyBinary(t, (d1, d2) -> d1 >= d2 ? 1.0 : 0.0);
    }

    public Tensor power(Tensor t) {
        return applyBinary(t, Math::pow);
    }

    public Tensor exponentiate() {
        return applyUnary(Math::exp);
    }
    
    public Tensor log() {
        return applyUnary(Math::log);
    }
    
    public Tensor negate() {
        return applyUnary(d -> -d);
    }
    
    public Tensor relu() {
        return applyUnary(d -> Math.max(0, d));
    }
    
    public Tensor sigmoid() {
        return applyUnary(d -> 1.0 / (1.0 + Math.exp(-d)));
    }
    
    public Tensor tanh() {
        return applyUnary(Math::tanh);
    }
    
    public abstract Tensor allSum();
    
    /**
     * The size of the dimension responsibly for training examples.
     * 
     * TODO: get a better interface later, but this is needed to prevent
     * breakages while generalizing to abstract Tensor
     * @return 
     */
    public abstract int mDim();
    
    public abstract double value(int... position);
    
    // TODO: It's probably desirable to implement shape with a different interface.
    public abstract String shape();
    
    public abstract Tensor applyUnary(Function<Double,Double> function);
    
    public abstract Tensor applyBinary(Tensor right, BiFunction<Double, Double, Double> function);
}
