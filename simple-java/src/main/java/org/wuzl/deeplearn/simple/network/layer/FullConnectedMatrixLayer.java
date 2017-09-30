package org.wuzl.deeplearn.simple.network.layer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.wuzl.deeplearn.simple.network.MatrixLayer;
import org.wuzl.deeplearn.simple.network.NetworkActivator;

/**
 * 全连接层实现类
 * 
 * @author ziliang.wu
 *
 */
public class FullConnectedMatrixLayer implements MatrixLayer {
	private final int inputSize;// 本层输入向量的维度
	private final int outputSize;// 本层输出向量的维度
	private final NetworkActivator activator;// 激活函数
	private final INDArray weightArray;// 权重数组
	private final INDArray offset;// 偏置项
	private INDArray output;// 输出
	private INDArray input;// 输入
	private INDArray delta;// 误差
	private INDArray weightGradient;// 权重梯度
	private INDArray offsetGradient;// 偏置项梯度

	public FullConnectedMatrixLayer(int inputSize, int outputSize, NetworkActivator activator) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.activator = activator;
		this.weightArray = Nd4j.rand(this.outputSize, inputSize, -0.1, 0.1, Nd4j.getRandom());
		this.offset = Nd4j.zeros(this.outputSize, 1);
		this.output = Nd4j.zeros(this.outputSize, 1);
	}

	@Override
	public void forward(INDArray inputArray) {
		if (inputArray.length() != inputSize) {
			throw new RuntimeException("输入向量的维度必须等于当前层的输入维度");
		}
		this.input = inputArray;
		this.output = activator.forward(weightArray.mmul(input).addi(offset));
	}

	@Override
	public void backward(INDArray deltaArray) {
		// 使用input是的当前层的值
		this.delta = activator.backward(this.input).mul(this.weightArray.transpose().mmul(deltaArray));
		this.weightGradient = deltaArray.mmul(this.input.transpose());
		this.offsetGradient = deltaArray.dup();
	}

	@Override
	public void update(double learningRate) {
		this.weightArray.addi(weightGradient.mul(learningRate));
		this.offset.addi(offsetGradient.mul(learningRate));
	}

	@Override
	public String toString() {
		return "FullConnectedMatrixLayer [weightArray=" + weightArray + ", offset=" + offset + "]";
	}

	@Override
	public INDArray getWeightArray() {
		return weightArray;
	}

	@Override
	public INDArray getOffset() {
		return offset;
	}

	@Override
	public INDArray getOutput() {
		return output;
	}

	@Override
	public INDArray getDelta() {
		return delta;
	}

	@Override
	public NetworkActivator getActivator() {
		return activator;
	}

	@Override
	public INDArray getWeightGradient() {
		return weightGradient;
	}

}
