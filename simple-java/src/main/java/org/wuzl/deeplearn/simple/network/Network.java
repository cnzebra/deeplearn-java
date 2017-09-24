package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Administrator
 *
 */
public class Network {
	private final List<NetworkLayer> layerList;// 神经网络的层集合
	private final List<NetwordConnection> connectionList = new ArrayList<NetwordConnection>();// 连接结合
	private final int layerCount;

	/**
	 * 默认层列表是排好序的 此处没有做处理
	 * 
	 * @param layerList
	 */
	public Network(List<Integer> layerNodeCountList) {
		if (layerNodeCountList == null || layerNodeCountList.isEmpty()) {
			throw new RuntimeException("神经网络的层不可以为空");
		}
		this.layerList = new ArrayList<>();
		for (int i = 0; i < layerNodeCountList.size(); i++) {
			layerList.add(new NetworkLayer(i, layerNodeCountList.get(i)));
		}
		this.layerCount = layerList.size();
		this.initConnecitonList();
	}

	/**
	 * 训练
	 * 
	 * @param inputs
	 *            训练样本特征。每个元素是一个样本的特征。
	 * @param labels
	 *            训练样本标签。每个元素是一个样本的标签。
	 * @param rate
	 *            频率
	 * @param iteration
	 *            训练总次数
	 */
	public void train(List<List<Double>> inputs, List<List<Double>> labels, double rate, int iteration) {
		if (inputs == null || labels == null || inputs.isEmpty()) {
			throw new RuntimeException("输入数据和结果不可以为空");
		}
		if (inputs.size() != labels.size()) {
			throw new RuntimeException("输入数据和结果数量不一致");
		}
		for (int i = 0; i < iteration; i++) {
			for (int j = 0; j < inputs.size(); j++) {
				predict(inputs.get(j));
				calcDelta(labels.get(j));
				updateWeight(rate);
			}
		}
	}

	/**
	 * 计算网络在一个样本下，每个连接上的梯度
	 * 
	 * @param input
	 * @param label
	 */
	public void calcGradient(List<Double> input, List<Double> label) {
		predict(input);
		calcDelta(label);
		calcGradient();
	}

	/**
	 * 根据输入的样本预测输出值
	 * 
	 * @param input
	 *            数组，样本的特征，也就是网络的输入向量
	 */
	public List<Double> predict(List<Double> input) {
		this.layerList.get(0).setOutput(input);
		for (int i = 1; i < layerList.size(); i++) {
			layerList.get(i).calcOutput();
		}
		List<Double> output = new ArrayList<>();
		NetworkLayer lastLayer = layerList.get(layerList.size() - 1);
		List<BaseNode> nodeList = lastLayer.getNodeList();
		for (int i = 0; i < nodeList.size() - 1; i++) {// 循环获取输出
			output.add(nodeList.get(i).getOutput());
		}
		return output;
	}

	public void print() {
		for (NetworkLayer layer : layerList) {
			layer.print();
		}
	}

	public List<NetwordConnection> getConnectionList() {
		return connectionList;
	}

	/**
	 * 计算每个节点的delta
	 * 
	 * @param label
	 */
	private void calcDelta(List<Double> label) {
		NetworkLayer lastLayer = layerList.get(layerList.size() - 1);
		List<BaseNode> lastLayerNodeList = lastLayer.getNodeList();
		// 计算输出层的误差
		for (int i = 0; i < label.size(); i++) {
			lastLayerNodeList.get(i).calcOutputLayerDelta(label.get(i));
		}
		// 计算隐藏层的误差
		for (int i = layerList.size() - 2; i >= 0; i--) {
			List<BaseNode> nodeList = layerList.get(i).getNodeList();
			for (BaseNode node : nodeList) {
				node.calcHiddenLayerDelta();
			}
		}
	}

	/**
	 * 更新每个连接权重
	 * 
	 * @param weight
	 */
	private void updateWeight(double rate) {
		for (int i = 0; i < layerList.size() - 1; i++) {
			List<BaseNode> nodeList = layerList.get(i).getNodeList();
			for (BaseNode node : nodeList) {
				for (NetwordConnection conneciton : node.downConnectionList) {
					conneciton.updateWeight(rate);
				}
			}
		}
	}

	/**
	 * 计算每个连接的梯度
	 */
	private void calcGradient() {
		for (int i = 0; i < layerList.size() - 1; i++) {
			List<BaseNode> nodeList = layerList.get(i).getNodeList();
			for (BaseNode node : nodeList) {
				for (NetwordConnection conneciton : node.downConnectionList) {
					conneciton.calcGradient();
				}
			}
		}
	}

	private void initConnecitonList() {
		for (int i = 0; i < layerCount - 1; i++) {
			NetworkLayer nowLayer = layerList.get(i);// 当前层
			NetworkLayer nextLayer = layerList.get(i+1);// 下一层
			List<BaseNode> nowNodeList = nowLayer.getNodeList();// 当前层节点
			List<BaseNode> nextNodeList = nextLayer.getNodeList();// 下一层节点
			for (int j = 0; j < nowNodeList.size(); j++) {
				for (int p = 0; p < nextNodeList.size() - 1; p++) {// 下一层节点连接
																	// 不需要 偏移量节点
					NetwordConnection connection = new NetwordConnection(nowNodeList.get(j), nextNodeList.get(p));
					connectionList.add(connection);
					connection.getUpNode().appendDownConnection(connection);
					connection.getDownNode().appendUpConnection(connection);
				}
			}
		}
	}

}
