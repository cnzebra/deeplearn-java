package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

/**
 * 负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作。
 * 
 * @author Administrator
 * 
 */
public class NetworkLayer {
	private final int layerIndex;// 层编号
	private final int nodeCount;// 节点总数
	private final List<BaseNode> nodeList;

	public NetworkLayer(int layerIndex, int nodeCount) {
		this.layerIndex = layerIndex;
		this.nodeCount = nodeCount;
		this.nodeList = new ArrayList<BaseNode>();
		for (int i = 0; i < nodeCount; i++) {
			nodeList.add(new NetworkNode(layerIndex, i));
		}
		// 加人偏置项节点
		nodeList.add(new ConstNode(layerIndex, nodeCount));
	}

	/**
	 * 设置层的各个节点的输出
	 * 
	 * @param output
	 */
	public void setOutput(List<Double> outputList) {
		if (outputList == null || outputList.isEmpty()) {
			throw new RuntimeException("输出列表不可以为空");
		}
		if (outputList.size() != nodeList.size()) {
			throw new RuntimeException("输出列表的个数必须等于节点的个数");
		}
		for (int i = 0; i < outputList.size(); i++) {
			nodeList.get(i).setOutput(outputList.get(i));
		}
	}

	/**
	 * 计算输出
	 */
	public void calcOutput() {
		for (BaseNode node : nodeList) {
			node.calcOut();
		}
	}

	public void dump() {
		for (BaseNode node : nodeList) {
			System.out.println(node);
		}
	}

	public int getLayerIndex() {
		return layerIndex;
	}

	public int getNodeCount() {
		return nodeCount;
	}

	public List<BaseNode> getNodeList() {
		return nodeList;
	}

}
