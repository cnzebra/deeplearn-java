package org.wuzl.deeplearn.simple.network;

import org.wuzl.deeplearn.simple.util.MathUtil;

/**
 * 节点之间的连接 主要职责是记录连接的权重，以及这个连接所关联的上下游节点。
 * 
 * @author ziliang.wu
 * 
 */
/**
 * @author Administrator
 *
 */
public class NetwordConnection {
	private final BaseNode upNode;// 前节点
	private final BaseNode downNode;// 后节点
	private double weight = MathUtil.random();// 权重
	private double gradient = 0.0;// 梯度

	public NetwordConnection(BaseNode upNode, BaseNode downNode) {
		this.upNode = upNode;
		this.downNode = downNode;
	}

	/**
	 * 计算梯度 =入参*误差 入参是前节点的输出 误差是下节点的误差
	 */
	public void calcGradient() {
		gradient = downNode.getDelta() * upNode.getOutput();
	}

	/**
	 * 根据梯度下降算法更新权重 weight=oldweight+rate*误差*入参
	 */
	public void updateWeight(double rate) {
		calcGradient();
		this.weight += (rate * gradient);
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public double getGradient() {
		return gradient;
	}

	public BaseNode getUpNode() {
		return upNode;
	}

	public BaseNode getDownNode() {
		return downNode;
	}

	public double getWeight() {
		return weight;
	}

	@Override
	public String toString() {
		return "NetwordConnection [upNode.layer_index=" + upNode.getLayerIndex() + ", upNode.node_index="
				+ upNode.getNodeIndex() + ", downNode.layer_index=" + downNode.getLayerIndex()
				+ ", downNode.node_index=" + downNode.getNodeIndex() + ", weight=" + weight + ", gradient=" + gradient
				+ "]";
	}

}
