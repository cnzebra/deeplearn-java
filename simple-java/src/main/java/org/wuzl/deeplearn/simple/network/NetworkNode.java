package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

import org.wuzl.deeplearn.simple.util.MathUtil;

/**
 * 节点对象计算和记录节点自身的信息(比如输出值、误差项等)，以及与这个节点相关的上下游的连接。
 * 
 * @author ziliang.wu
 * 
 */
public class NetworkNode extends BaseNode {

	private final List<NetwordConnection> upConnectionList = new ArrayList<NetwordConnection>();// 上游节点的连接列表

	public NetworkNode(int layerIndex, int nodeIndex) {
		super(layerIndex, nodeIndex);
	}

	// 设置节点输出
	public void setOutput(double output) {
		this.output = output;
	}

	/**
	 * 添加一个上游连接
	 * 
	 * @param connection
	 */
	public void appendUpConnection(NetwordConnection connection) {
		this.upConnectionList.add(connection);
	}

	/**
	 * 计算输出
	 */
	public void calcOut() {
		for (NetwordConnection connection : upConnectionList) {
			output += (connection.getUpNode().getOutput() * connection
					.getWeight());
		}
		this.output = MathUtil.sigmoid(output);
	}

	@Override
	public void calcHiddenLayerDelta() {
		double downCountDelta = 0.0;
		for (NetwordConnection connection : downConnectionList) {
			downCountDelta += (connection.getDownNode().getDelta() * connection
					.getWeight());
		}
		this.delta = output * (1 - output) * downCountDelta;
	}

	public double getOutput() {
		return output;
	}

	public double getDelta() {
		return delta;
	}

	@Override
	public String toString() {
		return "NetworkNode [layerIndex=" + layerIndex + ", nodeIndex="
				+ nodeIndex + ", downConnectionList=" + downConnectionList
				+ ", upConnectionList=" + upConnectionList + ", output="
				+ output + ", delta=" + delta + "]";
	}

	public int getLayerIndex() {
		return layerIndex;
	}

	public int getNodeIndex() {
		return nodeIndex;
	}

	public List<NetwordConnection> getDownConnectionList() {
		return downConnectionList;
	}

	public List<NetwordConnection> getUpConnectionList() {
		return upConnectionList;
	}

	public void setDelta(double delta) {
		this.delta = delta;
	}

}
