package org.wuzl.deeplearn.simple.network;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 以向量计算的Layer
 * 
 * @author ziliang.wu
 *
 */
public interface MatrixLayer {
	/**
	 * 前向计算
	 * 
	 * @param inputArray
	 *            输入向量，维度必须等于input_size
	 */
	public void forward(INDArray inputArray);

	/**
	 * 反向计算W和b的梯度
	 * 
	 * @param deltaArray从上一层传递过来的误差项
	 */
	public void backward(INDArray deltaArray);

	/**
	 * 使用梯度下降算法更新权重
	 * 
	 * @param learningRate
	 */
	public void update(double learningRate);

	public INDArray getOutput();

	public NetworkActivator getActivator();

	public INDArray getDelta();

	public INDArray getWeightArray();

	public INDArray getOffset();

	public INDArray getWeightGradient();
}
