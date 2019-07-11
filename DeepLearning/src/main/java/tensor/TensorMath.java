package tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;

/**
 * @author tyler
 */
public class TensorMath {
    
    /**
     * Apply a binary operation to two Tensors of possibly different shapes.
     * 
     * Supports numpy-inspired broadcasting.
     * 
     * @param left
     * @param right
     * @param function
     * @return The element-wise, broadcasted result function.apply(left, right).
     */
    public static Tensor applyBinary(Tensor left, Tensor right, BiFunction<Double, Double, Double> function) {
        var shape = getResultShape(left, right);
        
        // Handle differently, depending on the shape.
        if (shape.size() == 4) {
            // Tensor 4D
            int n0 = shape.get(0);
            int n1 = shape.get(1);
            int n2 = shape.get(2);
            int n3 = shape.get(3);
            double[][][][] data = new double[n0][n1][n2][n3];
            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    for (int k = 0; k < n2; k++) {
                        for (int l = 0; l < n3; l++) {
                           data[i][j][k][l] = function.apply(
                                   left.value2(i,j,k,l), 
                                   right.value2(i,j,k,l));
                        }
                    }
                }
            }
            return new Tensor4D(data);
        } 
        
        if (shape.size() == 2) {
            // Tensor 2D
            int n0 = shape.get(0);
            int n1 = shape.get(1);
            double[][] data = new double[n0][n1];
            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    data[i][j] = function.apply(
                            left.value2(i,j), 
                            right.value2(i,j));
                }
            }
            return new Tensor2D(data);
        }
        
        // Everything else is not a valid case right now.
        throw new UnsupportedOperationException("Shape: " + shape + " is currently unsupported.");
    }
    
    public static List<Integer> getResultShape(Tensor left, Tensor right) {
        // Fetch the shape of both Tensors.
        // Reverse the dimension lists since broadcasting considers last
        // dimension first.
        var shape1 = left.shape();
        Collections.reverse(shape1);
        var shape2 = right.shape();
        Collections.reverse(shape2);
        
        int s1 = shape1.size();
        int s2 = shape2.size();
        int max = Math.max(s1, s2);
        
        // Built the result shape, with last dimension first.
        var result = new ArrayList<Integer>();
        for (int i = 0; i < max; i++) {
            var dimLeft = 1;
            var dimRight = 1;
            if (i < s1) {
                dimLeft = shape1.get(i);
            }
            if (i < s2) {
                dimRight = shape2.get(i);
            }
            
            if (dimLeft == 1 || dimRight == 1 || dimLeft == dimRight) {
                // Binary operations are still ok
                result.add(Math.max(dimLeft, dimRight));
            } else {
                throw new IllegalArgumentException("Tensors are incompatible for binary operations.");
            }
        }
        
        // We built the result shape backwards, so revere before returning it.
        Collections.reverse(result);
        return result;
    }
}
