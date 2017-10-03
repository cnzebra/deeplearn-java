package org.wuzl.deeplearn.simple.cnn;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.wuzl.deeplearn.simple.cnn.activator.IdentityActivator;

public class ConvLayerTest {
	private static INDArray input;
	private static INDArray sensitivityArray;
	static {
		double[] flat = ArrayUtil.flattenDoubleArray(new double[][][] {
				{ { 0, 1, 1, 0, 2 }, { 2, 2, 2, 2, 1 }, { 1, 0, 0, 2, 0 }, { 0, 1, 1, 0, 0 }, { 1, 2, 0, 0, 2 } },
				{ { 1, 0, 2, 2, 0 }, { 0, 0, 0, 2, 0 }, { 1, 2, 1, 2, 1 }, { 1, 0, 0, 0, 0 }, { 1, 2, 1, 1, 1 } },
				{ { 2, 1, 2, 0, 0 }, { 1, 0, 0, 1, 0 }, { 0, 2, 1, 0, 1 }, { 0, 1, 2, 2, 2 }, { 2, 1, 0, 0, 1 } } });
		int[] shape = new int[] { 3, 5, 5 };
		input = Nd4j.create(flat, shape, 'c');
		flat = ArrayUtil.flattenDoubleArray(new double[][][] { { { 0, 1, 1 }, { 2, 2, 2 }, { 1, 0, 0 } },
				{ { 1, 0, 2 }, { 0, 0, 0 }, { 1, 2, 1 } } });
		shape = new int[] { 2, 3, 3 };
		sensitivityArray = Nd4j.create(flat, shape, 'c');
	}

	private static ConvLayer getConvLayer() {
		ConvLayer layer = new ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, 0.001, new IdentityActivator());
		List<Filter> filters = layer.getFilters();
		double[] flat = ArrayUtil.flattenDoubleArray(new double[][][] { { { -1, 1, 0 }, { 0, 1, 0 }, { 0, 1, 1 } },
				{ { -1, -1, 0 }, { 0, 0, 0 }, { 0, -1, 0 } }, { { 0, 0, -1 }, { 0, 1, 0 }, { 1, -1, -1 } } });
		int[] shape = new int[] { 3, 3, 3 };
		INDArray weights = Nd4j.create(flat, shape, 'c');
		filters.get(0).setWeights(weights);
		filters.get(0).setBias(1);
		flat = ArrayUtil.flattenDoubleArray(new double[][][] { { { 1, 1, -1 }, { -1, -1, 1 }, { 0, -1, 1 } },
				{ { 0, 1, 0 }, { -1, 0, -1 }, { -1, 1, 0 } }, { { -1, 0, 0 }, { -1, 0, 1 }, { -1, 0, 0 } } });
		weights = Nd4j.create(flat, shape, 'c');
		filters.get(1).setWeights(weights);
		return layer;
	}

	private static void gradientCheck() {
		ConvLayer layer = getConvLayer();
		layer.forward(input);
		// 求取sensitivity map
		INDArray sensitivityArray = Nd4j.ones(layer.getOutputArray().shape());
		// 计算梯度
		layer.backward(input, sensitivityArray, new IdentityActivator());
		// 检查梯度
		double epsilon = 10e-4;
		Filter filter = layer.getFilters().get(0);
		int[] gradientShape = filter.getWeightGradient().shape();
		INDArray weights = filter.getWeights();
		for (int d = 0; d < gradientShape[0]; d++) {
			for (int i = 0; i < gradientShape[1]; i++) {
				for (int j = 0; j < gradientShape[2]; j++) {
					int[] index = new int[] { d, i, j };
					weights.putScalar(index, weights.getDouble(index) + epsilon);
					layer.forward(input);
					double error1 = layer.getOutputArray().sumNumber().doubleValue();
					weights.putScalar(index, weights.getDouble(index) - 2 * epsilon);
					layer.forward(input);
					double error2 = layer.getOutputArray().sumNumber().doubleValue();
					double expectGrad = (error1 - error2) / (2 * epsilon);
					weights.putScalar(index, weights.getDouble(index) + epsilon);
					System.out.println("weight(" + d + "," + i + ",j):expected " + expectGrad + ",actural "
							+ filter.getWeightGradient().getDouble(index));
				}
			}
		}
	}

	private static void test() {
		ConvLayer layer = getConvLayer();
		layer.forward(input);
		System.out.println("输出");
		System.out.println(layer.getOutputArray());
	}

	private static void testBp() {
		ConvLayer layer = getConvLayer();
		layer.backward(input, sensitivityArray, new IdentityActivator());
		layer.update();
		System.out.println(layer.getFilters().get(0));
		System.out.println(layer.getFilters().get(1));
	}

	public static void main(String[] args) {
		// test();
		// testBp();
		gradientCheck();
	}
}
