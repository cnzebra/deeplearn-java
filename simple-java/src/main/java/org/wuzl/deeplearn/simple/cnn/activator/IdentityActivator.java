package org.wuzl.deeplearn.simple.cnn.activator;

import org.wuzl.deeplearn.simple.cnn.CnnActivator;

public class IdentityActivator implements CnnActivator {

	@Override
	public double forward(double input) {
		return input;
	}

	@Override
	public double backward(double output) {
		return 1;
	}

}
