package org.wuzl.deeplearn.simple;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestSetValue {
	public static void main(String[] args) {
		INDArray derivativeArray = Nd4j.rand(new int[] { 3, 4, 5 }, -100, 100, Nd4j.getRandom());
		System.out.println(derivativeArray);
		int[] shape = derivativeArray.shape();
		int num = 10;
		for (int d = 0; d < shape[0]; d++) {
			for (int i = 0; i < shape[1]; i++) {
				for (int j = 0; j < shape[2]; j++) {
					derivativeArray.putScalar(new int[] { d, i, j }, num++);
				}
			}

		}
		System.out.println(derivativeArray);
	}
}
