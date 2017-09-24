package org.wuzl.deeplearn.simple.mnist;

import java.util.Arrays;
import java.util.List;

import org.wuzl.deeplearn.simple.util.ByteUtil;
import org.wuzl.deeplearn.simple.util.TimeUtil;

import com.google.common.collect.Lists;

/**
 * 样本label数据加载器 数据是从第8个byte开始 一个label是1个字节
 * 
 * @author Administrator
 *
 */
public class MnistLabelLoader extends MnistLoader {
	public MnistLabelLoader(String path, int count) {
		super(path, count);
		buffer.position(8);// 丢弃非数据 文件的头
	}

	/**
	 * 转化为一个10维的one-hot向量
	 * 
	 * @param output
	 * @return
	 */
	private List<Double> norm(short output) {
		Double[] labelVec = new Double[10];
		Arrays.fill(labelVec, 0.1);
		labelVec[output] = 0.9;
		return Arrays.asList(labelVec);
	}

	/**
	 * 加载指定数量的label
	 * 
	 * @return
	 */
	public List<List<Double>> load() {
		System.out.println("MnistLabelLoader开始load，时间:" + TimeUtil.getNowTime());
		List<List<Double>> result = Lists.newArrayList();
		for (int i = 0; i < getCount(); i++) {
			result.add(norm(ByteUtil.getUnsigned(buffer.get())));
		}
		buffer.clear();
		System.out.println("MnistLabelLoader结束load，时间:" + TimeUtil.getNowTime());
		return result;
	}

	public static int getResult(List<Double> vec) {
		int maxValueIndex = 0;
		double maxValue = 0.0;
		for (int i = 0; i < vec.size(); i++) {
			if (vec.get(i) > maxValue) {
				maxValue = vec.get(i);
				maxValueIndex = i;
			}
		}
		return maxValueIndex;
	}

}
