package org.wuzl.deeplearn.simple.cnn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.wuzl.deeplearn.simple.cnn.util.CnnUtil;

/**
 * 取最大值的 下采样层
 * 
 * @author ziliang.wu
 *
 */
public class MaxPoolingLayer {
	private final int inputWidth;// 入参的宽度
	private final int inputHeight;// 入参的高度
	private final int channelNumber;// 入参通道数 也就是上一节点的filter数量 也可以认为是入参深度
	private final int filterWidth;// filter宽度
	private final int filterHeight;// filter高度
	private final int outputWidth;// 输出宽度
	private final int outputHeight;// 输出高度
	private final int stride;// 步幅
	private INDArray deltaArray;// 用于保存传递到上一层的sensitivity
								// map深度是入参的最外维度或者说上一层的fitler个数
	private INDArray outputArray;// 输出到下一层的

	public MaxPoolingLayer(int inputWidth, int inputHeight, int channelNumber, int filterWidth, int filterHeight,
			int stride) {
		this.inputWidth = inputWidth;
		this.inputHeight = inputHeight;
		this.channelNumber = channelNumber;
		this.filterWidth = filterWidth;
		this.filterHeight = filterHeight;
		this.stride = stride;
		this.outputWidth = (this.inputWidth - this.filterWidth) / stride + 1;
		this.outputHeight = (this.inputHeight - this.filterHeight) / stride + 1;
		this.outputArray = Nd4j.zeros(new int[] { this.channelNumber, this.outputHeight, this.outputWidth });
	}

	public void forward(INDArray input) {
		for (int d = 0; d < channelNumber; d++) {// 深度
			for (int i = 0; i < outputHeight; i++) {// 宽度
				for (int j = 0; j < outputWidth; j++) {// 高度
					INDArray selectArray = CnnUtil.getPatch(input.get(NDArrayIndex.point(d)), i, j, this.filterWidth,
							this.filterHeight, stride);
					outputArray.putScalar(new int[] { d, i, j }, selectArray.maxNumber().doubleValue());
				}
			}
		}
	}

	public void backward(INDArray input, INDArray sensitivityArray) {
		this.deltaArray = Nd4j.zeros(input.shape());
		for (int d = 0; d < channelNumber; d++) {// 深度
			for (int i = 0; i < outputHeight; i++) {// 宽度
				for (int j = 0; j < outputWidth; j++) {// 高度
					INDArray selectArray = CnnUtil.getPatch(input.get(NDArrayIndex.point(d)), i, j, this.filterWidth,
							this.filterHeight, stride);
					int[] index = CnnUtil.getMaxIndex(selectArray);// 当前是二维
					this.deltaArray.putScalar(new int[] { d, i * stride + index[0], j * stride + index[1] },
							sensitivityArray.getDouble(new int[] { d, i, j }));

				}
			}
		}
	}

	public INDArray getDeltaArray() {
		return deltaArray;
	}

	public INDArray getOutputArray() {
		return outputArray;
	}

}
