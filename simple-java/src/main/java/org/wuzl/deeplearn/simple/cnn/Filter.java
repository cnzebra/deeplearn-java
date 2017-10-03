package org.wuzl.deeplearn.simple.cnn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Filter类保存了卷积层的参数以及梯度，并且实现了用梯度下降算法来更新参数
 * 
 * @author ziliang.wu
 *
 */
public class Filter {
	private final int width;// 宽度
	private final int height;// 高度
	private final int depth;// 深度
	private INDArray weights;// 权重
	private INDArray weightGradient;// 权重梯度
	private double bias = 0;// 偏移量
	private double biasGradient = 0;// 偏移量梯度

	public Filter(int width, int height, int depth) {
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.weights = Nd4j.rand(new int[] { this.depth, this.height, this.width }, -0.0001, 0.0001, Nd4j.getRandom());
		this.weightGradient = Nd4j.zeros(new int[] { this.depth, this.height, this.width });
	}

	public void update(double learningRate) {
		this.weights = this.weights.sub(this.weightGradient.muli(learningRate));
		this.bias = this.bias - (this.biasGradient * learningRate);
	}

	public void setWeights(INDArray weights) {
		this.weights = weights;
	}

	public INDArray getWeights() {
		return weights;
	}

	public INDArray getWeightGradient() {
		return weightGradient;
	}

	public double getBias() {
		return bias;
	}

	public double getBiasGradient() {
		return biasGradient;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public void setBiasGradient(double biasGradient) {
		this.biasGradient = biasGradient;
	}

	@Override
	public String toString() {
		return "Filter [weights=" + weights + ", bias=" + bias + "]";
	}

}
