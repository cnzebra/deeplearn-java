package org.wuzl.deeplearn.simple.mnist;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.wuzl.deeplearn.simple.util.ByteUtil;
import org.wuzl.deeplearn.simple.util.TimeUtil;

import com.google.common.collect.Lists;

/**
 * 向量版本 样本label数据加载器 数据是从第8个byte开始 一个label是1个字节
 * 
 * @author Administrator
 *
 */
public class MatrixMnistLabelLoader extends MnistLoader {
	public MatrixMnistLabelLoader(String path, int count) {
		super(path, count);
		buffer.position(8);// 丢弃非数据 文件的头
	}

	/**
	 * 转化为一个10维的one-hot向量
	 * 
	 * @param output
	 * @return
	 */
	private INDArray norm(short output) {
		INDArray ndArray = Nd4j.zeros(10, 1).add(0.1);
		ndArray.putScalar(output, 0, 0.9);
		return ndArray;
	}

	/**
	 * 加载指定数量的label
	 * 
	 * @return
	 */
	public List<INDArray> load() {
		System.out.println("MnistLabelLoader开始load，时间:" + TimeUtil.getNowTime());
		List<INDArray> result = Lists.newArrayList();
		for (int i = 0; i < getCount(); i++) {
			result.add(norm(ByteUtil.getUnsigned(buffer.get())));
		}
		buffer.clear();
		System.out.println("MnistLabelLoader结束load，时间:" + TimeUtil.getNowTime());
		return result;
	}

	public static int getResult(INDArray vec) {
		int maxValueIndex = 0;
		double maxValue = 0.0;
		for (int i = 0; i < vec.rows(); i++) {
			double value = vec.getDouble(i, 0);
			if (value > maxValue) {
				maxValue = value;
				maxValueIndex = i;
			} else if (value < 0) {
				throw new RuntimeException("有负数出现");
			}
		}
		return maxValueIndex;
	}

}
