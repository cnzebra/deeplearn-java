package org.wuzl.deeplearn.simple.cnn.util;

import java.util.Arrays;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class CnnUtilTest {
	@Test
	public void shouldPadding() {
		INDArray input = Nd4j.create(new double[] { 10, 20, 40, 20, 50, 100 }, new int[] { 2, 3 });
		System.out.println(input);
		System.out.println(CnnUtil.padding(input, 0));
		System.out.println(CnnUtil.padding(input, 1));
		System.out.println(CnnUtil.padding(input, 2));
		System.out.println("三维");
		input = Nd4j.create(new double[] { 1, 2, 3, 2, 5, 10, 10, 20, 40, 20, 50, 100 }, new int[] { 2, 2, 3 });
		System.out.println(input);
		System.out.println("padding0");
		System.out.println(CnnUtil.padding(input, 0));
		System.out.println("padding1");
		System.out.println(CnnUtil.padding(input, 1));
		System.out.println("padding2");
		System.out.println(CnnUtil.padding(input, 2));
	}

	@Test
	public void shouldGetMaxIndex() {
		INDArray array = Nd4j.rand(new int[] { 3, 4 }, -100, 100, Nd4j.getRandom());
		System.out.println(array);
		System.out.println(Arrays.toString(CnnUtil.getMaxIndex(array)));
		array = Nd4j.rand(new int[] { 3, 4, 5 }, -100, 100, Nd4j.getRandom());
		System.out.println(array);
		System.out.println(array.maxNumber());
		System.out.println(Arrays.toString(CnnUtil.getMaxIndex(array)));
	}
}
