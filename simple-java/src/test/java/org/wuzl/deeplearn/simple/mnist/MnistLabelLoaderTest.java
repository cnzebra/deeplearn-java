package org.wuzl.deeplearn.simple.mnist;

import java.util.List;

public class MnistLabelLoaderTest {
//	@Test
	public void shouldGetVec() {
		MnistLabelLoader loader = new MnistLabelLoader("F:/data/MNIST/train-labels.idx1-ubyte", 10);
		List<List<Double>> result = loader.load();
		for(List<Double> vec:result) {
			System.out.println(vec);
			for(int i=0;i<vec.size();i++) {
				if(vec.get(i)==0.9) {
					System.out.println(i);
				}
			}
		}
	}
}
