package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

/**
 * 为了实现一个输出恒为1的节点(计算偏置项时需要) 所以没有上节点
 * 
 * @author Administrator
 * 
 */
public class ConstNode extends BaseNode {

	public ConstNode(int layerIndex, int nodeIndex) {
		super(layerIndex, nodeIndex);
	}

	

	/**
	 * 计算隐藏层误差 感觉有问题 恒等于0
	 */
	@Override
	public void calcHiddenLayerDelta() {
		double downCountDelta = 0.0;
		for (NetwordConnection connection : downConnectionList) {
			downCountDelta += (connection.getDownNode().getDelta() * connection
					.getWeight());
		}
		this.delta = output * (1 - output) * downCountDelta;
	}

	@Override
	public String toString() {
		return "ConstNode [layerIndex=" + layerIndex + ", nodeIndex="
				+ nodeIndex + ", output=" + output + ", downConnectionList="
				+ downConnectionList + ", delta=" + delta + "]";
	}

}
