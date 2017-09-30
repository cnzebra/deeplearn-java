package org.wuzl.deeplearn.simple.mnist;

import java.util.List;

import org.wuzl.deeplearn.simple.network.NetwordConnection;
import org.wuzl.deeplearn.simple.network.Network;
import org.wuzl.deeplearn.simple.util.TimeUtil;

import com.google.common.collect.Lists;

/**
 * 计算失败 网络梯度测试没问题但计算不成功 失败率太高
 * 
 * @author ziliang.wu
 *
 */
public class MnistMain {
	private final MnistImageLoader imageLoader;
	private final MnistLabelLoader labelLoader;
	private final MnistImageLoader testImageLoader;
	private final MnistLabelLoader testLabelLoader;
	private final List<List<Double>> testImageInput;
	private final List<List<Double>> testLabelList;
	// 三层网络
	private static final Network network = new Network(Lists.newArrayList(784, 100, 10));

	public MnistMain(int trainCount, int testCount) {
		this.imageLoader = new MnistImageLoader("D:/测试数据/MNIST/train-images.idx3-ubyte", trainCount);
		this.labelLoader = new MnistLabelLoader("D:/测试数据/MNIST/train-labels.idx1-ubyte", trainCount);
		this.testImageLoader = new MnistImageLoader("D:/测试数据/MNIST/t10k-images.idx3-ubyte", testCount);
		this.testLabelLoader = new MnistLabelLoader("D:/测试数据/MNIST/t10k-labels.idx1-ubyte", testCount);
		// this.imageLoader = new
		// MnistImageLoader("F:/data/MNIST/train-images.idx3-ubyte",
		// trainCount);
		// this.labelLoader = new
		// MnistLabelLoader("F:/data/MNIST/train-labels.idx1-ubyte",
		// trainCount);
		// this.testImageLoader = new
		// MnistImageLoader("F:/data/MNIST/t10k-images.idx3-ubyte", testCount);
		// this.testLabelLoader = new
		// MnistLabelLoader("F:/data/MNIST/t10k-labels.idx1-ubyte", testCount);

		testImageInput = testImageLoader.load();
		testLabelList = testLabelLoader.load();
	}

	/**
	 * 验证数据计算错误率
	 * 
	 * @return
	 */
	public double evaluate() {
		int error = 0;
		System.out.println("开始测试，当前时间:" + TimeUtil.getNowTime());
		for (int i = 0; i < testImageInput.size(); i++) {
			List<Double> input = testImageInput.get(i);
			int rightLabel = MnistLabelLoader.getResult(testLabelList.get(i));
			int predict = MnistLabelLoader.getResult(network.predict(input));
			if (rightLabel != predict) {
				error++;
			}

		}
		System.out.println("测试完毕，当前时间:" + TimeUtil.getNowTime());
		return (error + 0.0) / testImageInput.size();
	}

	public void trainAndEvaluate() {
		int epoch = 0;
		double lastErrorRatio = 1.0;
		List<List<Double>> imageInput = imageLoader.load();
		List<List<Double>> label = labelLoader.load();
		System.out.println("开始训练，当前时间:" + TimeUtil.getNowTime());
		while (true) {
			epoch++;
			network.train(imageInput, label, 0.01, 1);
			System.out.println("第" + epoch + "轮训练结束，当前时间:" + TimeUtil.getNowTime());
			if (epoch % 5 == 0) {
				double errorRatio = evaluate();
				System.out.println("第" + epoch + "轮训练错误率:" + errorRatio);
				// 当准确率开始下降时终止训练
				if (errorRatio > lastErrorRatio) {
					break;
				} else {
					lastErrorRatio = errorRatio;
				}
			}
		}
	}

	public static void main(String[] args) {
		long begin = System.currentTimeMillis();
		System.out.println("启动，时间：" + TimeUtil.getNowTime());
		MnistMain main = new MnistMain(60000, 10000);
		main.trainAndEvaluate();
		System.out.println(TimeUtil.getNowTime() + "训练结束,花费时间(秒):" + (System.currentTimeMillis() - begin) / 1000l);
		System.out.println("===============");
		for (NetwordConnection connection : network.getConnectionList()) {
			System.out.println(connection);
		}
		System.out.println("===============");
	}
}
