package tensor;

import java.text.DecimalFormat;
import java.util.List;
import java.util.function.Function;

/**
 * A basic 2D tensor implementation.
 *
 * TODO: consider adding a few more operations, such as concatenation or
 * subsetting negation trig functions get / set operations
 *
 * @author tyler
 */
public class Tensor2D extends Tensor {

    // TODO: probably remove nrows, ncols.
    public final int nrows;
    public final int ncols;
    private double[][] data;

    public Tensor2D(int nr, int nc) {
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
    public Tensor2D(double[][] inputData) {
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

    public Tensor2D(List<Integer> shape, double[] inputData) {
        super(shape);
        if (shape.size() != 2) {
            throw new IllegalArgumentException("Invalid dimensions for Tensor2D.");
        }

        nrows = shape.get(0);
        ncols = shape.get(1);
        
        double[][] data = new double[nrows][ncols];
        int position = 0;
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                data[i][j] = inputData[position];
                position++;
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
        int r = position[n - 2];
        int c = position[n - 1];
        return data[r % nrows][c % ncols];
    }

    public Tensor2D matrixMultiply(Tensor2D t) {
        if (this.ncols != t.nrows) {
            throw new IllegalArgumentException("Matrix Multiplication is not defined for matrices of shape "
                    + shape() + ", and " + t.shape() + ".");
        }
        Tensor2D result = new Tensor2D(nrows, t.ncols);
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
    public Tensor2D columnSum() {
        Tensor2D result = new Tensor2D(1, ncols);
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
    public Tensor2D rowSum() {
        Tensor2D result = new Tensor2D(nrows, 1);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                result.data[i][0] += this.data[i][j];
            }
        }

        return result;
    }

    public Tensor2D transpose() {
        Tensor2D result = new Tensor2D(ncols, nrows);
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

    public static Tensor2D one() {
        return constant(1.);
    }

    public static Tensor2D constant(double value) {
        double[][] data = {{value}};
        return new Tensor2D(data);
    }

    @Override
    public Tensor2D applyUnary(Function<Double, Double> function) {
        Tensor2D result = new Tensor2D(nrows, ncols);

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
            if (i != nrows - 1) {
                sb.append(System.lineSeparator());
            }
        }
        return sb.toString();
    }
}
