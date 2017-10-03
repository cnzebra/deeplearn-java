package org.wuzl.deeplearn.simple.cnn;

public interface CnnActivator {
	/**
	 * 前向计算
	 * 
	 * @param input
	 * @return
	 */
	public double forward(double input);

	/**
	 * 反向传播算法
	 * 
	 * @param output
	 *            实际是当前层的向量 上一层的输出
	 * @return
	 */
	public double backward(double output);
}
