package org.wuzl.deeplearn.simple.network;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.collect.Lists;

public class MatrixNetworkTest {
	@Test
	public void shouleCheckNetwork() {
		MatrixNetwork network = new MatrixNetwork(Lists.newArrayList(8, 3, 8));
		network.gradientCheck(Nd4j.rand(8, 1, -1, 1, Nd4j.getRandom()), Nd4j.rand(8, 1, -1, 1, Nd4j.getRandom()));
	}
}
