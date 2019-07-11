package tensor;

import java.text.DecimalFormat;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * 4D Tensor implementation.
 *
 * @author tyler
 */
public class Tensor4D extends Tensor {

    private double[][][][] data;

    public Tensor4D(int n0, int n1, int n2, int n3) {
        super(List.of(n0, n1, n2, n3));
        data = new double[n0][n1][n2][n3];
    }

    public Tensor4D(List<Integer> dimensions) {
        super(dimensions);
        if (dimensions.size() != 4) {
            throw new IllegalArgumentException("Invalid dimensions for Tensor4D.");
        }
        data = new double[dimensions.get(0)][dimensions.get(1)][dimensions.get(2)][dimensions.get(3)];
    }

    public Tensor4D(double[][][][] inputData) {
        super(List.of(inputData.length, inputData[0].length, inputData[0][0].length, inputData[0][0][0].length));
        data = new double[inputData.length][inputData[0].length][inputData[0][0].length][inputData[0][0][0].length];
        for (int i = 0; i < inputData.length; i++) {
            for (int j = 0; j < inputData[i].length; j++) {
                for (int k = 0; k < inputData[i][j].length; k++) {
                    for (int l = 0; l < inputData[i][j][k].length; l++) {
                        data[i][j][k][l] = inputData[i][j][k][l];
                    }
                }
            }
        }
    }

    @Override
    public Tensor allSum() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int mDim() {
        return shape().get(0);
    }

    @Override
    public double value(int... position) {
        if (position.length != 4) {
            throw new IllegalArgumentException("Invalid position for 4D Tensor");
        }
        return data[position[0]][position[1]][position[2]][position[3]];
    }

    @Override
    public double value2(int... position) {
        int n = position.length;
        var shape = shape();
        int i = position[n - 4];
        int j = position[n - 3];
        int k = position[n - 2];
        int l = position[n - 1];

        return data[i % shape.get(0)][j % shape.get(1)][k % shape.get(2)][l % shape.get(3)];
    }

    @Override
    public Tensor applyUnary(Function<Double, Double> function) {
        var dimensions = shape();
        Tensor4D result = new Tensor4D(dimensions);
        for (int i = 0; i < dimensions.get(0); i++) {
            for (int j = 0; i < dimensions.get(1); j++) {
                for (int k = 0; i < dimensions.get(2); k++) {
                    for (int l = 0; i < dimensions.get(3); l++) {
                        result.data[i][j][k][l] = function.apply(data[i][j][k][l]);
                    }
                }
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

        var shape = shape();
        for (int i = 0; i < shape.get(0); i++) {
            for (int j = 0; j < shape.get(1); j++) {
                sb.append("(")
                        .append(i)
                        .append(",")
                        .append(j)
                        .append(")")
                        .append(System.lineSeparator());
                for (int k = 0; k < shape.get(2); k++) {
                    for (int l = 0; l < shape.get(3); l++) {
                        sb.append(df.format(data[i][j][k][l]));
                        sb.append(" ");
                    }
                    sb.append(System.lineSeparator());
                }
            }
            if (i != shape.get(0) - 1) {
                sb.append(System.lineSeparator());
            }
        }
        return sb.toString();
    }
}
