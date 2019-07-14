package layer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import regularize.Regularizer;
import tensor.Tensor;
import tensor.Tensor4D;

/**
 * Implementation of a pooling neural network layer.
 *
 * Pooling layers are compatible only with 4D data for now.
 * 
 * TODO: write some code to test this implementation.
 *
 * @author tyler
 */
public class Pool implements Layer {

    public static final String ARGMAX = "argmax";
    public static final String OLD_ACTIVATION = "a_old";

    private final PoolingType type;
    private final List<Integer> size;
    private final List<Integer> stride;
    private final List<Integer> padding;

    public Pool(PoolingType type, List<Integer> size, List<Integer> stride, List<Integer> padding) {
        this.type = type;
        this.size = new ArrayList<>(size);
        this.stride = new ArrayList<>(stride);
        this.padding = new ArrayList<>(padding);
    }

    @Override
    public ForwardPropResult forwardPropagate(Tensor x, boolean training) {
        // Step 0: verify the result is 4D.
        var shape = x.shape();
        if (shape.size() != 4) {
            throw new IllegalArgumentException("Pool layer only supports Tensor4D currently.");
        }

        var cache = new HashMap<String, Tensor>();
        cache.put(OLD_ACTIVATION, x);

        // Step 1: compute the end shape.
        var m = shape.get(0);
        var in1 = shape.get(1);
        var in2 = shape.get(2);
        var in3 = shape.get(3);

        var out1 = getSize(in1, size.get(0), stride.get(0), padding.get(0));
        var out2 = getSize(in2, size.get(1), stride.get(1), padding.get(1));
        var out3 = getSize(in3, size.get(2), stride.get(2), padding.get(2));

        double[][][][] data = new double[m][out1][out2][out3];

        double factor = 1.0 / (size.get(0) * size.get(1) * size.get(2));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < out1; j++) {
                for (int k = 0; k < out2; k++) {
                    for (int l = 0; l < out3; l++) {
                        int a1 = Math.max(0, j * stride.get(0) - padding.get(0));
                        int a2 = Math.min(j * stride.get(0) - padding.get(0) + size.get(0), in1);
                        int b1 = Math.max(0, k * stride.get(1) - padding.get(1));
                        int b2 = Math.min(k * stride.get(1) - padding.get(1) + size.get(1), in2);
                        int c1 = Math.max(0, l * stride.get(2) - padding.get(2));
                        int c2 = Math.min(l * stride.get(2) - padding.get(2) + size.get(2), in3);

                        if (type == PoolingType.Max) {
                            // Find the max element in the 3D box.
                            for (int a = a1; a < a2; a++) {
                                for (int b = b1; b < b2; b++) {
                                    for (int c = c1; c < c2; c++) {
                                        data[i][j][k][l] = Math.max(data[i][j][k][l], x.value(i, a, b, c));
                                    }
                                }
                            }
                        } else if (type == PoolingType.Average) {
                            // Find the average element in the 3D box.
                            for (int a = a1; a < a2; a++) {
                                for (int b = b1; b < b2; b++) {
                                    for (int c = c1; c < c2; c++) {
                                        data[i][j][k][l] += x.value(i, a, b, c);
                                    }
                                }
                            }
                            data[i][j][k][l] *= factor;
                        }
                    }
                }
            }
        }

        return new ForwardPropResult(new Tensor4D(data), cache);
    }

    @Override
    public BackPropResult backwardPropagate(Tensor dA, ForwardPropResult cache) {
        var activation = cache.cache.get(OLD_ACTIVATION);
        var shape = activation.shape();
        var m = shape.get(0);
        var out1 = shape.get(1);
        var out2 = shape.get(2);
        var out3 = shape.get(3);

        var inputShape = dA.shape();
        var in1 = inputShape.get(1);
        var in2 = inputShape.get(2);
        var in3 = inputShape.get(3);

        double[][][][] data = new double[m][out1][out2][out3];
        double factor = 1.0 / (size.get(0) * size.get(1) * size.get(2));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < in1; j++) {
                for (int k = 0; k < in2; k++) {
                    for (int l = 0; l < in3; l++) {
                        int a1 = Math.max(0, j * stride.get(0) - padding.get(0));
                        int a2 = Math.min(j * stride.get(0) - padding.get(0) + size.get(0), out1);
                        int b1 = Math.max(0, k * stride.get(1) - padding.get(1));
                        int b2 = Math.min(k * stride.get(1) - padding.get(1) + size.get(1), out2);
                        int c1 = Math.max(0, l * stride.get(2) - padding.get(2));
                        int c2 = Math.min(l * stride.get(2) - padding.get(2) + size.get(2), out3);

                        if (type == PoolingType.Max) {
                            // Find the max element in the 3D box.
                            int[] arr = new int[3];
                            double max = Double.MIN_VALUE;
                            for (int a = a1; a < a2; a++) {
                                for (int b = b1; b < b2; b++) {
                                    for (int c = c1; c < c2; c++) {
                                        var value = activation.value(i, a, b, c);
                                        if (value > max) {
                                            max = value;
                                            arr = new int[]{a, b, c};
                                        }
                                    }
                                }
                            }
                            data[i][arr[0]][arr[1]][arr[2]] += dA.value(i, j, k, l);
                        } else if (type == PoolingType.Average) {
                            // Find the average element in the 3D box.
                            for (int a = a1; a < a2; a++) {
                                for (int b = b1; b < b2; b++) {
                                    for (int c = c1; c < c2; c++) {
                                        data[i][a][b][c] = dA.value(i, j, k, l) * factor;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return new BackPropResult(new Tensor4D(data), new HashMap<>());
    }

    @Override
    public void updateParameters(Map<String, Tensor> deltaParameters, Regularizer r) {
        // Pool only has hyperparameters.
        return;
    }

    @Override
    public String toString() {
        var sb = new StringBuilder();
        if (type == PoolingType.Max) {
            sb.append("Max Pool Layer");
        } else if (type == PoolingType.Average) {
            sb.append("Average Pool Layer");
        }
        sb.append("Size:");
        sb.append(System.lineSeparator());
        sb.append(size);
        sb.append(System.lineSeparator());
        sb.append("Stride:");
        sb.append(System.lineSeparator());
        sb.append(stride);
        sb.append(System.lineSeparator());
        sb.append("Padding:");
        sb.append(System.lineSeparator());
        sb.append(padding);

        return sb.toString();
    }

    private static int getSize(int originalSize, int filterSize, int stride, int padding) {
        return 1 + (originalSize + 2 * padding - filterSize) / stride;
    }
}
