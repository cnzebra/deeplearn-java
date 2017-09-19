package org.wuzl.deeplearn.simple;

import java.util.Arrays;
import java.util.List;

/**
 * Perceptron感知器
 * 
 * @author ziliang.wu
 * 
 */
public class Perceptron {
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
	public double predict(Double[] inputVecArray) {
		if (inputVecArray == null || inputVecArray.length != inputNum) {
			throw new RuntimeException("输入数据的个数不对");
		}
		double result = 0.0;
		for (int i = 0; i < inputVecArray.length; i++) {
			result += inputVecArray[i] * weights[i];
		}
		return activator.run(result + bias);
	}

	/**
	 * 训练
	 * 
	 * @param inputVecArrayList
	 *            一组向量
	 * @param labelList
	 *            每个向量对应的label
	 * @param iteration
	 *            训练轮数
	 * @param rate
	 *            学习率
	 */
	public void train(List<Double[]> inputVecArrayList, List<Double> labelList,
			int iteration, double rate) {
		if (inputVecArrayList == null || inputVecArrayList.isEmpty()
				|| labelList == null) {
			throw new RuntimeException("输入参数不全");
		}
		if (inputVecArrayList.size() != labelList.size()) {
			throw new RuntimeException("inputVecArrayList与labelList不匹配");
		}
		for (int i = 0; i < iteration; i++) {
			this.trainData(inputVecArrayList, labelList, rate);
		}
	}

	private void trainData(List<Double[]> inputVecArrayList,
			List<Double> labelList, double rate) {

		// 对每个样本，按照感知器规则更新权重
		for (int i = 0; i < inputVecArrayList.size(); i++) {
			Double[] inputVecArray = inputVecArrayList.get(i);
			if (inputVecArray.length != inputNum) {
				throw new RuntimeException("inputVecArray数量不对");
			}
			double label = labelList.get(i);
			// 计算感知器在当前权重下的输出
			double outPut = predict(inputVecArray);
			this.updateWeight(inputVecArray, outPut, label, rate);
		}
	}

	/**
	 * 利用感知器规则更新权重
	 * 
	 * @param inputVecArray
	 * @param outPut
	 * @param label
	 * @param rate
	 */
	private void updateWeight(Double[] inputVecArray, double outPut,
			double label, double rate) {
		double delta = label - outPut;
		bias += rate * delta;
		for (int i = 0; i < inputVecArray.length; i++) {
			weights[i] += rate * delta * inputVecArray[i];
		}
	}

}
