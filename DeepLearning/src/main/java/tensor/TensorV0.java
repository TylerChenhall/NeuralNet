package tensor;

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
public class TensorV0 {
    public final int nrows;
    public final int ncols;
    private double[][] data;
    
    public TensorV0(int nr, int nc) {
        if (nr <= 0 || nc <= 0) {
            throw new IllegalArgumentException("Tensor dimensions must be positive. Found (nr, nc) = (" +
                    nr + ", " + nc + ").");
        }
        nrows = nr;
        ncols = nc;
        data = new double[nr][nc];
    }
    
    /**
     * Construct TensorV0 from existing array.
     * 
     * @param inputData 
     */
    public TensorV0(double[][] inputData) {
        nrows = inputData.length;
        ncols = inputData[0].length;
        data = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++) {
            if (inputData[i].length != ncols) {
                throw new IllegalArgumentException("Tensor input array must be rectangular.");
            }
            for (int j = 0; j < ncols; j++) {
                data[i][j] = inputData[i][j];
            }
        }
    }
    
    public TensorV0 add(TensorV0 t) {
        TensorV0[] tensors = broadcastify(this, t);
        
        TensorV0 result = new TensorV0(tensors[0].nrows, tensors[0].ncols);
        
        for (int i = 0; i < result.nrows; i++) {
            for (int j = 0; j < result.ncols; j++) {
                result.data[i][j] = tensors[0].data[i][j] + tensors[1].data[i][j];
            }
        }
        
        return result;
    }

    public TensorV0 subtract(TensorV0 t) {
        TensorV0[] tensors = broadcastify(this, t);
        
        TensorV0 result = new TensorV0(tensors[0].nrows, tensors[0].ncols);
        
        for (int i = 0; i < result.nrows; i++) {
            for (int j = 0; j < result.ncols; j++) {
                result.data[i][j] = tensors[0].data[i][j] - tensors[1].data[i][j];
            }
        }
        
        return result;
    }

    public TensorV0 multiply(TensorV0 t) {
        TensorV0[] tensors = broadcastify(this, t);
        
        TensorV0 result = new TensorV0(tensors[0].nrows, tensors[0].ncols);
        
        for (int i = 0; i < result.nrows; i++) {
            for (int j = 0; j < result.ncols; j++) {
                result.data[i][j] = tensors[0].data[i][j] * tensors[1].data[i][j];
            }
        }
        
        return result;
    }

    public TensorV0 divideBy(TensorV0 t) {
        TensorV0[] tensors = broadcastify(this, t);
        
        TensorV0 result = new TensorV0(tensors[0].nrows, tensors[0].ncols);
        
        for (int i = 0; i < result.nrows; i++) {
            for (int j = 0; j < result.ncols; j++) {
                result.data[i][j] = tensors[0].data[i][j] / tensors[1].data[i][j];
            }
        }
        
        return result;
    }
    
    /**
     * Implements element-wise >= operator.
     * @param t
     * @return  The result is an indicator TensorV0, with 1.0 indicating true.
     */
    public TensorV0 atLeast(TensorV0 t) {
        TensorV0[] tensors = broadcastify(this, t);
        
        TensorV0 result = new TensorV0(tensors[0].nrows, tensors[0].ncols);
        
        for (int i = 0; i < result.nrows; i++) {
            for (int j = 0; j < result.ncols; j++) {
                result.data[i][j] = tensors[0].data[i][j] >= tensors[1].data[i][j] ? 1.0 : 0.0;
            }
        }
        
        return result;
    }

    public TensorV0 power(TensorV0 t) {
        TensorV0[] tensors = broadcastify(this, t);
        
        TensorV0 result = new TensorV0(tensors[0].nrows, tensors[0].ncols);
        
        for (int i = 0; i < result.nrows; i++) {
            for (int j = 0; j < result.ncols; j++) {
                result.data[i][j] = Math.pow(tensors[0].data[i][j], tensors[1].data[i][j]);
            }
        }
        
        return result;
    }
    
    public TensorV0 matrixMultiply(TensorV0 t) {
        if (this.ncols != t.nrows) {
            throw new IllegalArgumentException("Matrix Multiplication is not defined for matrices of shape " +
                    shape() + ", and " + t.shape() + ".");
        }
        TensorV0 result = new TensorV0(nrows, t.ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < t.ncols; j++) {
                for (int k = 0; k < ncols; k++) {
                    result.data[i][j] += data[i][k] * t.data[k][j];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Computes a Tensor by summing the values in each column.
     * 
     * @return A 1 x ncols TensorV0 of column sums
     */
    public TensorV0 columnSum() {
        TensorV0 result = new TensorV0(1, ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[0][j] += this.data[i][j];
            }
        }
        
        return result;
    }

    public TensorV0 exponentiate() {
        TensorV0 result = new TensorV0(nrows, ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[i][j] = Math.exp(this.data[i][j]);
            }
        }
        
        return result;
    }
    
    public TensorV0 log() {
        TensorV0 result = new TensorV0(nrows, ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[i][j] = Math.log(this.data[i][j]);
            }
        }
        
        return result;
    }
    
    public TensorV0 negate() {
        TensorV0 result = new TensorV0(nrows, ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[i][j] = -1.0 * this.data[i][j];
            }
        }
        
        return result;
    }
    
    public TensorV0 relu() {
        TensorV0 result = new TensorV0(nrows, ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[i][j] = Math.max(0., this.data[i][j]);
            }
        }
        
        return result;
    }
    
    /**
     * Computes a Tensor by summing the values in each row.
     * 
     * @return A nrows x 1 TensorV0 of row sums
     */
    public TensorV0 rowSum() {
        TensorV0 result = new TensorV0(nrows, 1);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[i][0] += this.data[i][j];
            }
        }
        
        return result;
    }
    
    public TensorV0 sigmoid() {
        return TensorV0.one().divideBy(
                TensorV0.one().add(this.negate().exponentiate())
        );
    }
    
    public TensorV0 tanh() {
        TensorV0 result = new TensorV0(nrows, ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[i][j] = Math.tanh(this.data[i][j]);
            }
        }
        
        return result;
    }

    public TensorV0 transpose() {
        TensorV0 result = new TensorV0(ncols, nrows);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        
        return result;
    }
    
    public String shape() {
        return "(" + nrows + ", " + ncols + ")";
    }
    
    public static TensorV0 one() {
        return constant(1.);
    }
    
    public static TensorV0 constant(double value) {
        double[][] data = {{value}};
        return new TensorV0(data);
    }
    
    /** 
     * Creates temporary broadcasted TensorV0s to ease element-wise operations.
     * 
     * This isn't the most efficient, but makes implementation simpler for now.
     * 
     * @param a
     * @param b
     * @return 
     */
    private static TensorV0[] broadcastify(TensorV0 a, TensorV0 b) {
        // First, check if the dimensions are compatible for element-wise ops
        // They're compatible if they have the same shape in each dimension or 1.
        boolean compatible = true;
        if (!(a.nrows == b.nrows || a.nrows == 1 || b.nrows == 1)) {
            compatible = false;
        }
        if (!(a.ncols == b.ncols || a.ncols == 1 || b.ncols == 1)) {
            compatible = false;
        }
        
        if (!compatible) {
            throw new IllegalArgumentException("Input Tensors are not compatible for element-wise operations.");
        }
        
        // Prepare for broadcasting by extending the rows and columns as needed.
        if (a.nrows == 1 && b.nrows != 1) {
            a = rowBroadcast(a, b.nrows);
        } else if (a.nrows != 1 && b.nrows == 1) {
            b = rowBroadcast(b, a.nrows);
        }
        if (a.ncols == 1 && b.ncols != 1) {
            a = colBroadcast(a, b.ncols);
        } else if (a.ncols != 1 && b.ncols == 1) {
            b = colBroadcast(b, a.ncols);
        }
        
        TensorV0[] tensors = {a, b};
        return tensors;
    }
    
    /** 
     * Copies the given row vector nrows times.
     * @param input
     * @param nrows
     * @return 
     */
    private static TensorV0 rowBroadcast(TensorV0 input, int nrows) {
        TensorV0 result = new TensorV0(nrows, input.ncols);
        for (int i = 0; i < result.nrows; i++) {
            for (int j = 0; j < result.ncols; j++) {
                result.data[i][j] = input.data[0][j];
            }
        }
        
        return result;
    }
    
    /**
     * Copies the given column vector ncols times.
     * @param input
     * @param ncols
     * @return 
     */
    private static TensorV0 colBroadcast(TensorV0 input, int ncols) {
        TensorV0 result = new TensorV0(input.nrows, ncols);
        for (int i = 0; i < result.nrows; i++) {
            for (int j = 0; j < result.ncols; j++) {
                result.data[i][j] = input.data[i][0];
            }
        }
        
        return result;
    }
}