package org.wuzl.deeplearn.simple.mnist;

import java.util.List;

import org.wuzl.deeplearn.simple.util.ByteUtil;
import org.wuzl.deeplearn.simple.util.TimeUtil;

import com.google.common.collect.Lists;

/**
 * 图像数据加载器 数据是从第16个byte开始 一个图片是28*28字节
 * 
 * @author Administrator
 *
 */
public class MnistImageLoader extends MnistLoader {
	public MnistImageLoader(String path, int count) {
		super(path, count);
		buffer.position(16);// 丢弃非图片数据 文件的头
	}

	/**
	 * 获取一个图片文件这种只能 单线程处理
	 * 
	 * @return
	 */
	private List<Double> getPicture() {
		
		byte[] img = new byte[28 * 28];
		buffer.get(img);
		List<Double> result=Lists.newArrayList();
		for (int i = 0; i < img.length; i++) {
			result.add(ByteUtil.getUnsignedDouble(img[i]));
		}
		return result;
	}

	/**
	 * 加载指定数量的图片
	 * 
	 * @return
	 */
	public List<List<Double>> load() {
		System.out.println("MnistImageLoader开始load，时间:"+TimeUtil.getNowTime());
		List<List<Double>> result = Lists.newArrayList();
		for (int i = 0; i < getCount(); i++) {
			result.add(getPicture());
		}
		buffer.clear();
		System.out.println("MnistImageLoader结束load，时间:"+TimeUtil.getNowTime());
		return result;
	}

}
