package tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/** 
 * A generic Tensor interface.
 * 
 * @author tyler
 */
public abstract class Tensor {
    private final List<Integer> dimensions;
    
    public Tensor(List<Integer> dimensions) {
        for (var dim : dimensions) {
            if (dim <= 0) {
                throw new IllegalArgumentException("Tensor dimensions must be positive.");
            }
        }
        this.dimensions = new ArrayList<>(dimensions);
    }
    
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
    
    public Tensor abs() {
        return applyUnary(Math::abs);
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
    public int mDim() {
        return dimensions.get(0);
    }
    
    /**
     * Return the value in the specified position of the Tensor.
     * 
     * Position arguments must be valid (within range, correct dimensionality).
     * 
     * @param position
     * @return 
     */
    public abstract double value(int... position);
    
    /**
     * Return a value, even if position arguments don't fit the data object.
     * 
     * Essentially, this means two things:
     *   Apply the modulus operator when position indices are too big
     *   Ignore extra position indices
     * 
     * To mesh well with broadcasting, early indices are ignored if there are
     * too many.
     * 
     * @param position
     * @return 
     */
    public abstract double value2(int... position);
    
    public List<Integer> shape() {
        return new ArrayList<>(dimensions);
    }
    
    public abstract Tensor applyUnary(Function<Double,Double> function);
    
    public Tensor applyBinary(Tensor right, BiFunction<Double, Double, Double> function) {
        return TensorMath.applyBinary(this, right, function);
    }
}
