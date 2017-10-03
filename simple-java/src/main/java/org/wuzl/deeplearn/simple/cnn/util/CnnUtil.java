package org.wuzl.deeplearn.simple.cnn.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class CnnUtil {
	/**
	 * 为数组增加Zero padding，自动适配输入为2D和3D的情况
	 * 
	 * @param input
	 * @param zeroPaddingNum
	 * @return
	 */
	public static INDArray padding(INDArray input, int zeroPaddingNum) {
		if (zeroPaddingNum == 0) {
			return input;
		}
		int rank = input.rank();
		if (rank == 2) {// 二维
			int width = input.columns();
			int height = input.rows();
			INDArray paddingArray = Nd4j.zeros(height + 2 * zeroPaddingNum, width + 2 * zeroPaddingNum);
			for (int i = zeroPaddingNum; i < zeroPaddingNum + input.rows(); i++) {
				for (int j = zeroPaddingNum; j < zeroPaddingNum + input.columns(); j++) {
					paddingArray.putScalar(i, j, input.getDouble(i - zeroPaddingNum, j - zeroPaddingNum));
				}
			}
			return paddingArray;
		} else if (rank == 3) {// 三维
			int[] shape = input.shape();
			int depth = shape[0];
			int height = shape[1];
			int width = shape[2];
			INDArray paddingArray = Nd4j.zeros(depth, height + 2 * zeroPaddingNum, width + 2 * zeroPaddingNum);
			for (int d = 0; d < depth; d++) {
				for (int i = zeroPaddingNum; i < zeroPaddingNum + height; i++) {
					for (int j = zeroPaddingNum; j < zeroPaddingNum + width; j++) {
						paddingArray.putScalar(d, i, j, input.getDouble(d, i - zeroPaddingNum, j - zeroPaddingNum));
					}
				}
			}
			return paddingArray;
		}
		throw new RuntimeException("维数不正确,当前维数:" + rank);
	}

	/**
	 * 从输入数组中获取本次卷积的区域， 自动适配输入为2D和3D的情况
	 * 
	 * @param inputArray
	 * @param i
	 * @param j
	 * @param filterWidth
	 * @param filterHeight
	 * @param stride
	 * @return
	 */
	public static INDArray getPatch(INDArray inputArray, int i, int j, int filterWidth, int filterHeight, int stride) {
		int startI = i * stride;
		int startJ = j * stride;
		if (inputArray.rank() == 2) {
			return inputArray.get(NDArrayIndex.interval(startI, startI + filterHeight),
					NDArrayIndex.interval(startJ, startJ + filterWidth));
		} else if (inputArray.rank() == 3) {
			return inputArray.get(NDArrayIndex.all(), NDArrayIndex.interval(startI, startI + filterHeight),
					NDArrayIndex.interval(startJ, startJ + filterWidth));
		}
		throw new RuntimeException("不支持的维度:" + inputArray.rank());
	}

	public static int[] getMaxIndex(INDArray inputArray) {
		int maxIndexValue = Nd4j.getExecutioner().execAndReturn(new IMax(inputArray)).getFinalResult();
		int[] shape = inputArray.shape();
		int[] result = new int[shape.length];
		for (int i = shape.length - 1; i >= 0; i--) {
			int shapeValue = shape[i];
			result[i] = maxIndexValue % shapeValue;
			maxIndexValue = maxIndexValue / shapeValue;
		}
		return result;
	}
}
