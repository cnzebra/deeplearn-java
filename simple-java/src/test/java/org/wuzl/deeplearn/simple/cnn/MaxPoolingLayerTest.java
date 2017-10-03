package org.wuzl.deeplearn.simple.cnn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

public class MaxPoolingLayerTest {
	private static INDArray input;
	private static INDArray sensitivityArray;
	static {
		double[] flat = ArrayUtil.flattenDoubleArray(new double[][][] {
				{ { 1.0, 1.0, 2.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 3.0, 2.0, 1.0, 0.0 }, { 1.0, 2.0, 3.0, 4.0 } },
				{ { 0.0, 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0, 7.0 }, { 8.0, 9.0, 0.0, 1.0 }, { 3.0, 4.0, 5.0, 6.0 } } });
		int[] shape = new int[] { 2, 4, 4 };
		input = Nd4j.create(flat, shape, 'c');
		flat = ArrayUtil.flattenDoubleArray(
				new double[][][] { { { 1.0, 2.0 }, { 2.0, 4.0 } }, { { 3.0, 5.0 }, { 8.0, 2.0 } } });
		shape = new int[] { 2, 2, 2 };
		sensitivityArray = Nd4j.create(flat, shape, 'c');
	}

	public static void main(String[] args) {
		MaxPoolingLayer layer = new MaxPoolingLayer(4, 4, 2, 2, 2, 2);
		System.out.println("测试forward");
		layer.forward(input);
		System.out.println("input array:");
		System.out.println(input);
		System.out.println("output array:");
		System.out.println(layer.getOutputArray());
		System.out.println("测试backward");
		layer = new MaxPoolingLayer(4, 4, 2, 2, 2, 2);
		layer.backward(input, sensitivityArray);
		System.out.println("input array:");
		System.out.println(input);
		System.out.println("sensitivity array:");
		System.out.println(sensitivityArray);
		System.out.println("delta array:");
		System.out.println(layer.getDeltaArray());

	}

}
