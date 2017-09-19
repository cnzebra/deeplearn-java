package org.wuzl.deeplearn.simple;

import java.util.Arrays;

/**
 * Perceptron感知器
 * 
 * @author ziliang.wu
 *
 */
public class Perceptron<T, E> {
	private final Activator activator;// 激活函数
	private final int inputNum;// 输入参数数量
	private double bias;// 偏移量
	double[] weights;// 权重

	public Perceptron(Activator activator, int inputNum) {
		this.activator = activator;
		this.inputNum = inputNum;
		weights = new double[inputNum];
		for (int i = 0; i < inputNum; i++) {
			weights[i] = 0.0;
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("weights :").append(Arrays.toString(weights));
		sb.append("\nbias\t:").append(bias);
		return sb.toString();
	}

	/**
	 * 输入向量，输出感知器的计算结果
	 * 
	 * @param inputVecArray
	 * @return
	 */
	private double predict(double[] inputVecArray) {
		if (inputVecArray == null || inputVecArray.length != inputNum) {
			throw new RuntimeException("输入数据的个数不对");
		}
		double result = 0.0;
		for (int i = 0; i < inputVecArray.length; i++) {
			result += inputVecArray[i] * weights[i];
		}
		return activator.run(result + bias);
	}
}
