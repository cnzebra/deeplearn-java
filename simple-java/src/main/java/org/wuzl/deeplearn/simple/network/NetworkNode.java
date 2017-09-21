package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

/**
 * 节点对象计算和记录节点自身的信息(比如输出值、误差项等)，以及与这个节点相关的上下游的连接。
 * 
 * @author ziliang.wu
 *
 */
public class NetworkNode {
	private final int layerIndex;// 节点所属的层编号
	private final int nodeIndex;// 节点的编号
	private final List<NetwordConnection> downConnectionList = new ArrayList<NetwordConnection>();// 下游节点的连接列表
	private final List<NetwordConnection> upConnectionList = new ArrayList<NetwordConnection>();// 上游节点的连接列表
	private double output;// 节点的输出值
	private double delta;// 偏移量

	public NetworkNode(int layerIndex, int nodeIndex) {
		this.layerIndex = layerIndex;
		this.nodeIndex = nodeIndex;
	}

	// 设置节点输出
	public void setOutput(double output) {
		this.output = output;
	}

	/**
	 * 添加一个到下游节点的连接
	 * 
	 * @param connection
	 */
	public void appendDownConnection(NetwordConnection connection) {
		this.downConnectionList.add(connection);
	}

	/**
	 * 添加一个上游连接
	 * 
	 * @param connection
	 */
	public void appendUpConnection(NetwordConnection connection) {
		this.upConnectionList.add(connection);
	}
	
}
