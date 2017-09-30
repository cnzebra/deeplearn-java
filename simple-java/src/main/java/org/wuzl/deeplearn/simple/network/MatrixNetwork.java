package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.wuzl.deeplearn.simple.network.activator.SigmoidActivator;
import org.wuzl.deeplearn.simple.network.layer.FullConnectedMatrixLayer;

import com.google.common.collect.Lists;

/**
 * 使用向量的网络
 * 
 * @author ziliang.wu
 *
 */
public class MatrixNetwork {
	private final List<MatrixLayer> layerList;

	public MatrixNetwork(List<Integer> layerInputList) {
		layerList = new ArrayList<>();
		SigmoidActivator activator = new SigmoidActivator();
		for (int i = 0; i < layerInputList.size() - 1; i++) {
			layerList.add(new FullConnectedMatrixLayer(layerInputList.get(i), layerInputList.get(i + 1), activator));
		}
	}

	/**
	 * 使用神经网络实现预测
	 * 
	 * @param sampleInput
	 *            输入样本
	 * @return
	 */
	public INDArray predict(INDArray sampleInput) {
		INDArray output = sampleInput;
		for (MatrixLayer layer : layerList) {
			layer.forward(output);
			output = layer.getOutput();
		}
		return output;
	}

	/**
	 * 训练
	 * 
	 * @param labels
	 *            样本标签
	 * @param inputList
	 *            输入样本
	 * @param rate
	 *            学习速率
	 * @param epoch
	 *            训练轮数
	 */
	public void train(List<INDArray> labels, List<INDArray> inputList, double rate, int epoch) {
		if (labels == null || inputList == null || labels.isEmpty()) {
			throw new RuntimeException("样本标签和输入样本数据不可以为空");
		}
		if (labels.size() != inputList.size()) {
			throw new RuntimeException("样本标签和输入样本数据总数不一致");
		}
		for (int i = 0; i < epoch; i++) {
			for (int j = 0; j < inputList.size(); j++) {
				this.trainOneTime(labels.get(j), inputList.get(j), rate);
			}
		}
	}

	/**
	 * 梯度检查
	 * 
	 * @param label
	 * @param input
	 */
	public void gradientCheck(INDArray label, INDArray input) {
		// 获取网络在当前样本下每个连接的梯度
		predict(input);
		calcGradient(label);
		double epsilon = 0.001;
		for (MatrixLayer layer : layerList) {
			INDArray weightArray = layer.getWeightArray();
			for (int i = 0; i < weightArray.rows(); i++) {
				for (int j = 0; j < weightArray.columns(); j++) {
					weightArray.putScalar(i, j, weightArray.getDouble(i, j) + epsilon);
					INDArray output = predict(input);
					double error1 = lose(output, label);
					weightArray.putScalar(i, j, weightArray.getDouble(i, j) - epsilon * 2);
					output = predict(input);
					double error2 = lose(output, label);
					double errorGradient = (error1 - error2) / (2 * epsilon);
					weightArray.putScalar(i, j, weightArray.getDouble(i, j) + epsilon);
					System.out.println("weights(" + i + "," + j + ")    expected gradient:" + errorGradient
							+ "\n\t\tactural  gradient:" + layer.getWeightGradient().getDouble(i, j));
				}
			}
		}

	}

	private void trainOneTime(INDArray label, INDArray input, double rate) {
		this.predict(input);
		this.calcGradient(label);
		this.updateWeight(rate);
	}

	private INDArray calcGradient(INDArray label) {
		MatrixLayer lastLayer = layerList.get(layerList.size() - 1);
		INDArray delta = lastLayer.getActivator().backward(lastLayer.getOutput())
				.mul(label.sub(lastLayer.getOutput()));
		for (int i = (layerList.size() - 1); i >= 0; i--) {
			MatrixLayer layer = layerList.get(i);
			layer.backward(delta);
			delta = layer.getDelta();
		}
		return delta;
	}

	private void updateWeight(double rate) {
		for (MatrixLayer layer : layerList) {
			layer.update(rate);
		}
	}

	/**
	 * 计算错误
	 * 
	 * @param output
	 * @param label
	 * @return
	 */
	private double lose(INDArray output, INDArray label) {
		return 0.5 * ((label.sub(output).mul(label.sub(output))).sumNumber().doubleValue());
	}

	public void dump() {
		for (MatrixLayer layer : layerList) {
			System.out.println(layer);
		}
	}

	public static void main(String[] args) {
		MatrixNetwork network = new MatrixNetwork(Lists.newArrayList(8, 3, 8));
		network.gradientCheck(Nd4j.rand(8, 1, 0, 10, Nd4j.getRandom()), Nd4j.rand(8, 1, 0, 10, Nd4j.getRandom()));
	}
}
