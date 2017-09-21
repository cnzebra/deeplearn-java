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

}
