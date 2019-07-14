package layer;

import java.util.HashMap;
import java.util.Map;
import regularize.Regularizer;
import tensor.Tensor;
import tensor.Tensor2D;
import tensor.Tensor4D;

/**
 * Flattens the input Tensor from 4D to 2D.
 * 
 * Dimension 0, which represents the number of examples, is preserved.
 * 
 * One application of flatten is prior to applying FullyConnected layers on
 * data which is originally 4D.
 * 
 * @author tyler
 */
public class Flatten implements Layer {
    
    public static final String ORIGINAL_SHAPE = "original_shape";

    @Override
    public ForwardPropResult forwardPropagate(Tensor x, boolean training) {
        var shape = x.shape();
        if (shape.size() != 4) {
            throw new IllegalArgumentException("Flatten only supports Tensor4D input.");
        }
        
        int m = shape.get(0);
        int n1 = shape.get(1);
        int n2 = shape.get(2);
        int n3 = shape.get(3);
        double[][] data = new double[m][n1 * n2 * n3];
        
        for (int i = 0; i < m; i++) {
            int index = 0;
            for (int j = 0; j < n1; j++) {
                for (int k = 0; k < n2; k++) {
                    for (int l = 0; l < n3; l++) {
                        data[i][index] = x.value(i,j,k,l);
                        index++;
                    }
                }
            }
        }
        
        var cache = new HashMap<String, Tensor>();
        double[][] shapeData = new double[][]{{m,n1,n2,n3}};
        cache.put(ORIGINAL_SHAPE, new Tensor2D(shapeData));
        
        return new ForwardPropResult(new Tensor2D(data), cache);
    }

    @Override
    public BackPropResult backwardPropagate(Tensor dA, ForwardPropResult cache) {
        var shape = dA.shape();
        if (shape.size() != 2) {
            throw new IllegalArgumentException("Flatten backpropagation input should be Tensor2D.");
        }
        
        // To backpropagate, we just reshape dA.
        var originalShape = cache.cache.get(ORIGINAL_SHAPE);
        int m = (int) originalShape.value(0,0);
        int n1 = (int) originalShape.value(0,1);
        int n2 = (int) originalShape.value(0,2);
        int n3 = (int) originalShape.value(0,3);
        
        double[][][][] data = new double[m][n1][n2][n3];
        for (int i = 0; i < m; i++) {
            int index = 0;
            for (int j = 0; j < n1; j++) {
                for (int k = 0; k < n2; k++) {
                    for (int l = 0; l < n3; l++) {
                        data[i][j][k][l] = dA.value(i, index);
                        index++;
                    }
                }
            }
        }
        
        return new BackPropResult(new Tensor4D(data), new HashMap<>());
    }

    @Override
    public void updateParameters(Map<String, Tensor> deltaParameters, Regularizer r) {
        // Flatten layer has no parameters.
        return;
    }
    
}
