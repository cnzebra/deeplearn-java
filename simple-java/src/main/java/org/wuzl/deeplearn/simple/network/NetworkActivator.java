package org.wuzl.deeplearn.simple.network;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 网络的激活函数
 * 
 * @author ziliang.wu
 *
 */
public interface NetworkActivator {
	/**
	 * 前向计算
	 * 
	 * @param ndArray
	 * @return
	 */
	public INDArray forward(INDArray ndArray);

	/**
	 * 反向传播算法
	 * 
	 * @param out
	 *            实际是当前层的向量 上一层的输出
	 * @return
	 */
	public INDArray backward(INDArray out);
}
