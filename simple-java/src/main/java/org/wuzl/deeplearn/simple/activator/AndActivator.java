package org.wuzl.deeplearn.simple.activator;

import org.wuzl.deeplearn.simple.Activator;

/**
 * 实现and的激活函数
 * 
 * @author Administrator
 * 
 */
public class AndActivator implements Activator {

	@Override
	public double run(double output) {
		return output > 0 ? 1 : 0;
	}

}
