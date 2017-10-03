package org.wuzl.deeplearn.simple.cnn.activator;

import org.wuzl.deeplearn.simple.cnn.CnnActivator;

public class ReluActivator implements CnnActivator {

	@Override
	public double forward(double input) {
		return input > 0 ? input : 0;
	}

	@Override
	public double backward(double output) {
		return output > 0 ? 1 : 0;
	}

}
