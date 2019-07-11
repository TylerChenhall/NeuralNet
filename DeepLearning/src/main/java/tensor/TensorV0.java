package tensor;

import java.text.DecimalFormat;
import java.util.List;
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
public class TensorV0 extends Tensor {
    // TODO: probably remove nrows, ncols.
    public final int nrows;
    public final int ncols;
    private double[][] data;
    
    public TensorV0(int nr, int nc) {
        super(List.of(nr, nc));
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
        super(List.of(inputData.length, inputData[0].length));
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
    
    @Override
    public double value(int... position) {
        if (position.length != 2) {
            throw new IllegalArgumentException("Invalid position for 2D Tensor");
        }
        return data[position[0]][position[1]];
    }
    
    @Override
    public double value2(int... position) {
        int n = position.length;
        int r = position[n-2];
        int c = position[n-1];
        return data[r % nrows][c % ncols];
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

    public TensorV0 transpose() {
        TensorV0 result = new TensorV0(ncols, nrows);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        
        return result;
    }
    
    @Override
    public Tensor allSum() {
        return this.rowSum().columnSum();
    }
    
    /**
     * mDim implementation.
     * 
     * TODO: I think m should really be the first dimension (nrows). However, this
     * would require me to make some changes first.
     * @return 
     */
    @Override
    public int mDim() {
        return ncols;
    }
    
    public static TensorV0 one() {
        return constant(1.);
    }
    
    public static TensorV0 constant(double value) {
        double[][] data = {{value}};
        return new TensorV0(data);
    }
    
    @Override
    public TensorV0 applyUnary(Function<Double,Double> function) {
        TensorV0 result = new TensorV0(nrows, ncols);
        
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[i][j] = function.apply(data[i][j]);
            }
        }
        
        return result;
    }
    
    @Override
    public String toString() {
        return toString(3);
    }

    public String toString(int decimalPrecision) {
        StringBuilder sb = new StringBuilder();
        DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(decimalPrecision);
        df.setMinimumFractionDigits(decimalPrecision);
        
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                sb.append(df.format(data[i][j]));
                sb.append(" ");
            }
            if (i != nrows-1) {
                sb.append(System.lineSeparator());
            }
        }
        return sb.toString();
    }
}
