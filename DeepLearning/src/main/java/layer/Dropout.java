package layer;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import regularize.Regularizer;
import tensor.Tensor;
import tensor.TensorBuilder;

/**
 * Implements a Dropout layer for a fixed dropout probability.
 * 
 * @author tyler
 */
public class Dropout implements Layer {
    public static final String MASK = "mask";
    
    private final double probability;
    private final Random random;
    
    public Dropout(double probability, long seed) {
        this.probability = probability;
        random = new Random(seed);
    }
    
    public Dropout(double probability) {
        this.probability = probability;
        random = new Random(TensorBuilder.DEFAULT_SEED);
    }

    @Override
    public ForwardPropResult forwardPropagate(Tensor x, boolean training) {
        // During training, apply dropout. Otherwise, do nothing.
        if (!training) {
            return new ForwardPropResult(x, null);
        }
        
        var inputShape = x.shape();
        int elementCount = inputShape.stream().reduce(0, Integer::sum);
        double[] data = new double[elementCount];
        double multiplier = 1.0 / probability;
        for (int i = 0; i < elementCount; i++) {
            data[i] = random.nextDouble() < probability ? multiplier : 0.0;
        }
        
        var mask = TensorBuilder.buildFromShapeAndValues(inputShape, data);
        var result = x.multiply(mask);
        
        var cache = new HashMap<String, Tensor>();
        cache.put(MASK, mask);
        
        return new ForwardPropResult(result, cache);
    }

    @Override
    public BackPropResult backwardPropagate(Tensor dA, ForwardPropResult cache) {
        var mask = cache.cache.get(MASK);
        var daPrev = dA.multiply(mask);
        
        return new BackPropResult(daPrev, new HashMap<>());
        
    }

    @Override
    public void updateParameters(Map<String, Tensor> deltaParameters, Regularizer r) {
        // There are no parameters to update.
        return;
    }
    
}
