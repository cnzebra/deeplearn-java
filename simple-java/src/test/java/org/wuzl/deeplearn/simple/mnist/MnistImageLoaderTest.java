package org.wuzl.deeplearn.simple.mnist;

import java.util.List;

import org.junit.Test;

public class MnistImageLoaderTest {
//	 @Test
	public void shouldGetImg() {
		MnistImageLoader loader = new MnistImageLoader("F:/data/MNIST/train-images.idx3-ubyte", 1);
		List<List<Double>> result = loader.load();
		System.out.println(result.get(0));
	}
}
