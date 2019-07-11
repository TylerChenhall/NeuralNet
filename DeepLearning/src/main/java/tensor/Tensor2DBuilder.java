package tensor;

import java.util.Random;

/**
 * Builds Tensors with values determined by various initialization strategies.
 * 
 * @author tyler
 */
public class Tensor2DBuilder {
    public static final long DEFAULT_SEED = 2019L;
    
    public static Tensor2D heInitialization(int nrows, int ncols, long seed) {
        double standardDeviation = Math.sqrt(2.0 / ncols);
        
        return sdInitialization(nrows, ncols, standardDeviation, seed);
    }
    
    public static Tensor2D heInitialization(int nrows, int ncols) {
        return heInitialization(nrows, ncols, DEFAULT_SEED);
    }
    
    public static Tensor2D xavierInitialization(int nrows, int ncols, long seed) {
        double standardDeviation = Math.sqrt(1.0 / ncols);
        
        return sdInitialization(nrows, ncols, standardDeviation, seed);
    }
    
    public static Tensor2D xavierInitialization(int nrows, int ncols) {
        return xavierInitialization(nrows, ncols, DEFAULT_SEED);
    }
    
    public static Tensor2D sdInitialization(int nrows, int ncols, double standardDeviation, long seed) {
        double[][] data = new double[nrows][ncols];
        Random r = new Random(seed);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                data[i][j] = r.nextGaussian() * standardDeviation;
            }
        }
        
        return new Tensor2D(data);
    }
}
