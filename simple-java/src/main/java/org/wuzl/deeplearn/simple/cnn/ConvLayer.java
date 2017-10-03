package org.wuzl.deeplearn.simple.cnn;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.wuzl.deeplearn.simple.cnn.util.CnnUtil;

/**
 * 卷积层
 * 
 * @author ziliang.wu
 *
 */
/**
 * @author ziliang.wu
 *
 */
public class ConvLayer {
	private final int inputWidth;// 入参的宽度
	private final int inputHeight;// 入参的高度
	private final int channelNumber;// 入参通道数 也就是上一节点的filter数量 也可以认为是入参深度
	private final int filterWidth;// filter宽度
	private final int filterHeight;// filter高度
	private final int filterNumber;// filter个数 也就是当前层的深度
	private final int zeroPadding;// 0填充的行列数
	private final int stride;// 步幅
	private final double learningRate;// 学习率
	private final CnnActivator activator;// 激活函数
	private final int outputWidth;// 输出宽度
	private final int outputHeight;// 输出高度
	private final List<Filter> filters = new ArrayList<>();// filter的个数就是输出的个数
															// 和深度
	private INDArray inputArray;// 输入
	private INDArray paddingInputArray;// 补零后的输入
	private INDArray outputArray;// 输出到下一层的
	private INDArray deltaArray;// 用于保存传递到上一层的sensitivity map
								// 深度是入参的最外维度或者说上一层的fitler个数

	public ConvLayer(int inputWidth, int inputHeight, int channelNumber, int filterWidth, int filterHeight,
			int filterNumber, int zeroPadding, int stride, double learningRate, CnnActivator activator) {
		this.inputWidth = inputWidth;
		this.inputHeight = inputHeight;
		this.channelNumber = channelNumber;
		this.filterWidth = filterWidth;
		this.filterHeight = filterHeight;
		this.filterNumber = filterNumber;
		this.zeroPadding = zeroPadding;
		this.stride = stride;
		this.learningRate = learningRate;
		this.activator = activator;
		this.outputWidth = calculateOutputSize(this.inputWidth, this.filterWidth, this.zeroPadding, this.stride);
		this.outputHeight = calculateOutputSize(this.inputWidth, this.filterHeight, this.zeroPadding, this.stride);
		this.outputArray = Nd4j.zeros(new int[] { this.filterNumber, this.outputHeight, this.outputWidth });
		for (int i = 0; i < this.filterNumber; i++) {
			this.filters.add(new Filter(filterWidth, filterHeight, this.channelNumber));
		}
	}

	/**
	 * 计算卷积层的输出
	 * 
	 * @param input
	 */
	public void forward(INDArray input) {
		this.inputArray = input;
		this.paddingInputArray = CnnUtil.padding(input, this.zeroPadding);
		for (int i = 0; i < filters.size(); i++) {
			Filter filter = filters.get(i);
			this.conv(this.paddingInputArray, filter.getWeights(), this.outputArray.get(NDArrayIndex.point(i)),
					this.stride, filter.getBias(), Op.FORWARD, this.activator);
		}
	}

	/**
	 * 计算传递给前一层的误差项，以及计算每个权重的梯度
	 * 
	 * @param input
	 * @param sensitivityArray
	 *            本层的sensitivity map
	 * @param activator
	 *            上一层的激活函数
	 */
	public void backward(INDArray input, INDArray sensitivityArray, CnnActivator activator) {
		this.forward(input);
		this.bpSensitivityMap(sensitivityArray, activator);
		this.bpGradient(sensitivityArray);
	}

	/**
	 * 按照梯度下降，更新权重
	 */
	public void update() {
		for (Filter filter : filters) {
			filter.update(this.learningRate);
		}
	}

	public INDArray getOutputArray() {
		return outputArray;
	}

	public void setOutputArray(INDArray outputArray) {
		this.outputArray = outputArray;
	}

	public INDArray getDeltaArray() {
		return deltaArray;
	}

	public void setDeltaArray(INDArray deltaArray) {
		this.deltaArray = deltaArray;
	}

	public List<Filter> getFilters() {
		return filters;
	}

	/**
	 * 反向传播误差 计算传递到上一层的sensitivity map
	 * 
	 * @param sensitivityArray
	 *            本层的sensitivity map
	 * @param activator
	 *            上一层的激活函数
	 */
	private void bpSensitivityMap(INDArray sensitivityArray, CnnActivator activator) {
		// 处理卷积步长，对原始sensitivity map进行扩展
		INDArray expandArray = expandSensitivityMap(sensitivityArray);
		// full卷积，对sensitivitiy map进行zero padding 方便后边使用公式
		int expandedWidth = expandArray.shape()[2];
		int zeroPadding = (this.inputWidth + this.filterWidth - 1 - expandedWidth) / 2;
		INDArray paddingArray = CnnUtil.padding(expandArray, zeroPadding);
		// 初始化deltaArray，用于保存传递到上一层的 sensitivity map
		this.deltaArray = this.createDeltaArray();
		// 对于具有多个filter的卷积层来说，最终传递到上一层的 sensitivity map相当于所有的filter的sensitivity
		// map之和
		for (int filterIndex = 0; filterIndex < filters.size(); filterIndex++) {
			Filter filter = filters.get(filterIndex);
			INDArray flippedWeights = Transforms.reverse(filter.getWeights(), true);// 三维的
			// 计算与一个filter对应的delta_array
			INDArray nowDeltaArray = createDeltaArray();
			for (int d = 0; d < this.channelNumber; d++) {
				// 误差 最外维度与filter个数一直 nowDeltaArray是上一层误差 最外层维度weights的最外维度一致
				conv(paddingArray.get(NDArrayIndex.point(filterIndex)), flippedWeights.get(NDArrayIndex.point(d)),
						nowDeltaArray.get(NDArrayIndex.point(d)), 1, 0, Op.NO_OP, null);
			}
			this.deltaArray = this.deltaArray.addi(nowDeltaArray);
		}
		// 将计算结果与激活函数的偏导数做element-wise乘法操作
		INDArray derivativeArray = this.inputArray.dup();
		int[] shape = derivativeArray.shape();
		for (int d = 0; d < shape[0]; d++) {
			for (int i = 0; i < shape[1]; i++) {
				for (int j = 0; j < shape[2]; j++) {
					int[] indexes = new int[] { d, i, j };
					derivativeArray.putScalar(indexes, activator.backward(derivativeArray.getDouble(indexes)));
				}
			}

		}
		this.deltaArray = this.deltaArray.muli(derivativeArray);
	}

	/**
	 * 生成空的上级误差
	 * 
	 * @return
	 */
	private INDArray createDeltaArray() {
		return Nd4j.zeros(new int[] { this.channelNumber, this.inputHeight, this.inputWidth });
	}

	/**
	 * 反向传播梯度
	 * 
	 * @param sensitivityArray
	 */
	private void bpGradient(INDArray sensitivityArray) {
		// 处理卷积步长，对原始sensitivity map进行扩展
		INDArray expandArray = expandSensitivityMap(sensitivityArray);
		for (int filterIndex = 0; filterIndex < filters.size(); filterIndex++) {
			Filter filter = filters.get(filterIndex);
			int weightsDep = filter.getWeights().shape()[0];// 权重的深度
			// 计算每个权重的梯度
			for (int d = 0; d < weightsDep; d++) {
				conv(this.paddingInputArray.get(NDArrayIndex.point(d)),
						expandArray.get(NDArrayIndex.point(filterIndex)),
						filter.getWeightGradient().get(NDArrayIndex.point(d)), 1, 0, Op.BACKWARD, null);
			}
			// 计算偏置项的梯度
			filter.setBiasGradient(expandArray.get(NDArrayIndex.point(filterIndex)).sumNumber().doubleValue());
		}
	}

	/**
	 * 将步长为S的sensitivity map『还原』为步长为1的sensitivity map
	 * 
	 * @param sensitivityArray
	 */
	private INDArray expandSensitivityMap(INDArray sensitivityArray) {
		int[] shape = sensitivityArray.shape();
		// 计算stride为1时sensitivity map的大小
		int expandedWidth = this.inputWidth - this.filterWidth + 2 * this.zeroPadding + 1;
		int expandedHeight = this.inputHeight - this.filterHeight + 2 * this.zeroPadding + 1;
		// 构建新的sensitivity_map
		INDArray expandArray = Nd4j.zeros(new int[] { shape[0], expandedHeight, expandedWidth });
		// 从原始sensitivity map拷贝误差值 输出值是跟误差高宽相等的 深度也一样
		for (int i = 0; i < this.outputHeight; i++) {
			int iPos = i * stride;
			for (int j = 0; j < this.outputWidth; j++) {
				int jPos = j * stride;
				for (int depth = 0; depth < shape[0]; depth++) {
					expandArray.putScalar(new int[] { depth, iPos, jPos }, sensitivityArray.getDouble(depth, i, j));
				}

			}
		}
		return expandArray;
	}

	/**
	 * 计算某个filter的卷积，自动适配输入为2D和3D的情况
	 * 
	 * @param nowInputArray
	 * @param nowKernelArray
	 * @param nowOutputArray
	 *            会更改这个属性
	 * @param stride
	 * @param bias
	 * @param op
	 * @param filter
	 */
	private void conv(INDArray nowInputArray, INDArray nowKernelArray, INDArray nowOutputArray, int stride, double bias,
			Op op, CnnActivator activator) {
		int outputWidth = nowOutputArray.columns();
		int outputHeight = nowOutputArray.rows();
		int[] shape = nowKernelArray.shape();
		int kernelWidth = shape[shape.length - 1];
		int kernelHeight = shape[shape.length - 2];
		for (int i = 0; i < outputHeight; i++) {
			for (int j = 0; j < outputWidth; j++) {
				double output;
				// 如果输入是二维的
//				if (nowInputArray.rank() == 2) {
//					// 取权重的第一维第一个数据
//					output = (CnnUtil.getPatch(nowInputArray, i, j, kernelWidth, kernelHeight, stride)
//							.mul(nowKernelArray.get(NDArrayIndex.point(0))).sumNumber().doubleValue()) + bias;
//				} else {
//					output = (CnnUtil.getPatch(nowInputArray, i, j, kernelWidth, kernelHeight, stride)
//							.mul(nowKernelArray).sumNumber().doubleValue()) + bias;
//				}
				output = (CnnUtil.getPatch(nowInputArray, i, j, kernelWidth, kernelHeight, stride)
						.mul(nowKernelArray).sumNumber().doubleValue()) + bias;
				if (op == Op.FORWARD) {
					// 调用激活函数
					output = activator.forward(output);
				} else if (op == Op.BACKWARD) {

				}
				nowOutputArray.put(i, j, output);
			}
		}

	}

	/**
	 * 计算输出的宽度和高度
	 * 
	 * @param inputSize
	 * @param filterSize
	 * @param zeroPadding
	 * @param stride
	 * @return
	 */
	private int calculateOutputSize(int inputSize, int filterSize, int zeroPadding, int stride) {
		return (inputSize - filterSize + 2 * zeroPadding) / stride + 1;
	}

	enum Op {
		FORWARD, BACKWARD, NO_OP
	}
}
