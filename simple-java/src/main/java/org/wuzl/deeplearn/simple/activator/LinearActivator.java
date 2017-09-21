package org.wuzl.deeplearn.simple.activator;

import org.wuzl.deeplearn.simple.Activator;

/**
 * 线性单元激活函数
 * 
 * @author ziliang.wu
 *
 */
public class LinearActivator implements Activator {

	@Override
	public double run(double input) {
		return input;
	}

}
