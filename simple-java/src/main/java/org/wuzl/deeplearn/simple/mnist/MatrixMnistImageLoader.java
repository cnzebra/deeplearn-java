package org.wuzl.deeplearn.simple.mnist;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.wuzl.deeplearn.simple.util.ByteUtil;
import org.wuzl.deeplearn.simple.util.TimeUtil;

import com.google.common.collect.Lists;

/**
 * 向量版本 图像数据加载器 数据是从第16个byte开始 一个图片是28*28字节
 * 
 * @author Administrator
 *
 */
public class MatrixMnistImageLoader extends MnistLoader {
	public MatrixMnistImageLoader(String path, int count) {
		super(path, count);
		buffer.position(16);// 丢弃非图片数据 文件的头
	}

	/**
	 * 获取一个图片文件这种只能 单线程处理
	 * 
	 * @return
	 */
	private INDArray getPicture() {

		byte[] img = new byte[28 * 28];
		buffer.get(img);
		INDArray ndArray = Nd4j.zeros(img.length, 1);
		for (int i = 0; i < img.length; i++) {
			ndArray.putScalar(i, 0, ByteUtil.getUnsignedDouble(img[i]));
		}
		return ndArray;
	}

	/**
	 * 加载指定数量的图片
	 * 
	 * @return
	 */
	public List<INDArray> load() {
		System.out.println("MnistImageLoader开始load，时间:" + TimeUtil.getNowTime());
		List<INDArray> result = Lists.newArrayList();
		for (int i = 0; i < getCount(); i++) {
			result.add(getPicture());
		}
		buffer.clear();
		System.out.println("MnistImageLoader结束load，时间:" + TimeUtil.getNowTime());
		return result;
	}
}
