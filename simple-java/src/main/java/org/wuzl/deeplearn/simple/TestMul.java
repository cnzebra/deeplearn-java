package org.wuzl.deeplearn.simple;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestMul {
	public static void main(String[] args) {
		INDArray arr1 = Nd4j.create(new double[] { 1, 3, 2, 5 }, new int[] { 2, 2 });

		INDArray arr2 = Nd4j.ones(new int[] { 1, 2, 2 }).add(9);
		System.out.println(arr1);
		System.out.println(arr2);
	}
}
