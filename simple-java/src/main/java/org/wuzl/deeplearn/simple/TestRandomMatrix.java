package org.wuzl.deeplearn.simple;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class TestRandomMatrix {
	public static void main(String[] args) {
		System.out.println(Nd4j.randn(3, 2));

		System.out.println(Nd4j.rand(3, 2));// 0-1
		System.out.println(Nd4j.rand(3, 2, 0, 256, Nd4j.getRandom()));
		INDArray array = Nd4j.rand(new int[] { 3, 4, 5 }, -100, 100, Nd4j.getRandom());
		System.out.println(array);
		// System.out.println(array.rank());
		array.get(NDArrayIndex.point(0)).put(1, 2, 250);
		System.out.println(array.get(NDArrayIndex.point(0)));
		System.out.println(">>>>>>>>>");
//		System.out.println(array.get(NDArrayIndex.point(0)));
//		System.out.println(array.getScalar(1, 1, 1));
		// System.out.println(array.get(NDArrayIndex.point(1)));

		System.out.println(array);
		// System.out.println(">");
		// System.out.println(array.get(NDArrayIndex.all(),NDArrayIndex.interval(1,1+2),NDArrayIndex.interval(1,1+2)));
	}
}
