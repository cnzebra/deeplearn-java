package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

public abstract class BaseNode {
	protected final int layerIndex;// 节点所属的层编号
	protected final int nodeIndex;// 节点的编号
	protected final List<NetwordConnection> downConnectionList = new ArrayList<NetwordConnection>();// 下游节点的连接列表
	protected double output = 0.0;// 节点的输出值
	protected double delta;// 误差

	public BaseNode(int layerIndex, int nodeIndex) {
		super();
		this.layerIndex = layerIndex;
		this.nodeIndex = nodeIndex;
	}

	/**
	 * 添加一个到下游节点的连接
	 * 
	 * @param connection
	 */
	public void appendDownConnection(NetwordConnection connection) {
		this.downConnectionList.add(connection);
	}

	public void setOutput(double output) {
		this.output = output;
	}

	public int getLayerIndex() {
		return layerIndex;
	}

	public int getNodeIndex() {
		return nodeIndex;
	}

	public double getOutput() {
		return output;
	}

	public double getDelta() {
		return delta;
	}

	/**
	 * 计算隐藏层误差
	 */
	public abstract void calcHiddenLayerDelta();

	/**
	 * 计算输出
	 */
	public abstract void calcOut();

	/**
	 * 添加下一节点
	 * 
	 * @param connection
	 */
	public abstract void appendUpConnection(NetwordConnection connection);
}
