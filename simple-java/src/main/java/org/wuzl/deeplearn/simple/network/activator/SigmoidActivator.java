package org.wuzl.deeplearn.simple.network.activator;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.factory.Nd4j;
import org.wuzl.deeplearn.simple.network.NetworkActivator;

public class SigmoidActivator implements NetworkActivator {

	@Override
	public INDArray forward(INDArray ndArray) {
		// 1.0 / (1.0 + np.exp(-weighted_input))

		INDArray expInd = Nd4j.getExecutioner().execAndReturn(new Exp(ndArray.mul(-1)));
		expInd.addi(1.0);
		// expInd = Nd4j.ones(2, 2).divi(expInd);//用除法
		// Nd4j.getExecutioner().execAndReturn(new Pow(expInd, -1));// 用pow
		expInd.rdivi(1);// 用倒数
		return expInd;
	}

	@Override
	public INDArray backward(INDArray out) {
		return out.mul(out.rsub(1));
	}
}
