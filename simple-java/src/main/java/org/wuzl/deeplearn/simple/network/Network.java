package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

public class Network {
	private final List<NetworkLayer> layerList;// 神经网络的层集合
	private final List<NetwordConnection> connectionList = new ArrayList<NetwordConnection>();// 连接结合
	private final int layerCount;

	/**
	 * 默认层列表是排好序的 此处没有做处理
	 * 
	 * @param layerList
	 */
	public Network(List<NetworkLayer> layerList) {
		if (layerList == null || layerList.isEmpty()) {
			throw new RuntimeException("神经网络的层不可以为空");
		}
		this.layerList = layerList;
		this.layerCount = layerList.size();
		this.initConnecitonList();
	}

	private void initConnecitonList() {
		for (int i = 0; i < layerCount - 1; i++) {
			NetworkLayer nowLayer = layerList.get(i);// 当前层
			NetworkLayer nextLayer = layerList.get(i++);// 下一层
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
