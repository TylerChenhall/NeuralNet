package tensor;

import java.util.Random;

/**
 * Builds Tensors with values determined by various initialization strategies.
 * 
 * @author tyler
 */
public class TensorV0Builder {
    public static final long DEFAULT_SEED = 2019L;
    
    public static TensorV0 heInitialization(int nrows, int ncols, long seed) {
        double standardDeviation = Math.sqrt(2.0 / ncols);
        
        return sdInitialization(nrows, ncols, standardDeviation, seed);
    }
    
    public static TensorV0 heInitialization(int nrows, int ncols) {
        return heInitialization(nrows, ncols, DEFAULT_SEED);
    }
    
    public static TensorV0 xavierInitialization(int nrows, int ncols, long seed) {
        double standardDeviation = Math.sqrt(1.0 / ncols);
        
        return sdInitialization(nrows, ncols, standardDeviation, seed);
    }
    
    public static TensorV0 xavierInitialization(int nrows, int ncols) {
        return xavierInitialization(nrows, ncols, DEFAULT_SEED);
    }
    
    public static TensorV0 sdInitialization(int nrows, int ncols, double standardDeviation, long seed) {
        double[][] data = new double[nrows][ncols];
        Random r = new Random(seed);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                data[i][j] = r.nextGaussian() * standardDeviation;
            }
        }
        
        return new TensorV0(data);
    }
}
