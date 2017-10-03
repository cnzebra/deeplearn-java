package org.wuzl.deeplearn.simple;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TestRot {
	public static void main(String[] args) {
		INDArray array = Nd4j.rand(new int[] { 2, 2 }, -100, 100, Nd4j.getRandom());
		System.out.println(array);
		System.out.println(Transforms.reverse(array, true));

	}
}
