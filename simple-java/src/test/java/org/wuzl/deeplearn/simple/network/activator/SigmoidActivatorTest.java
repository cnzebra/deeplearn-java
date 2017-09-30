package org.wuzl.deeplearn.simple.network.activator;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SigmoidActivatorTest {
	private SigmoidActivator activator = new SigmoidActivator();

	@Test
	public void shoulForward() {
		INDArray ndArray = Nd4j.ones(2, 2);
		ndArray.addi(-0.1);
		System.out.println(activator.forward(ndArray));
	}

	@Test
	public void shouldBackward() {
		INDArray ndArray = Nd4j.ones(2, 2);
		ndArray.addi(5);
		System.out.println(activator.backward(ndArray));
	}
}
