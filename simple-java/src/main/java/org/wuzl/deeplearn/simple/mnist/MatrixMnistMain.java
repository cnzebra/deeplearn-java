package org.wuzl.deeplearn.simple.mnist;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.wuzl.deeplearn.simple.network.MatrixNetwork;
import org.wuzl.deeplearn.simple.util.TimeUtil;

import com.google.common.collect.Lists;

/**
 * 向量版本 网络梯度测试没问题但计算不成功 失败率太高
 * 
 * @author ziliang.wu
 *
 */
public class MatrixMnistMain {
	private final MatrixMnistImageLoader imageLoader;
	private final MatrixMnistLabelLoader labelLoader;
	private final MatrixMnistImageLoader testImageLoader;
	private final MatrixMnistLabelLoader testLabelLoader;
	private final List<INDArray> testImageInput;
	private final List<INDArray> testLabelList;
	// 三层网络
	private static final MatrixNetwork network = new MatrixNetwork(Lists.newArrayList(784, 100, 10));

	public MatrixMnistMain(int trainCount, int testCount) {
		this.imageLoader = new MatrixMnistImageLoader("D:/测试数据/MNIST/train-images.idx3-ubyte", trainCount);
		this.labelLoader = new MatrixMnistLabelLoader("D:/测试数据/MNIST/train-labels.idx1-ubyte", trainCount);
		this.testImageLoader = new MatrixMnistImageLoader("D:/测试数据/MNIST/t10k-images.idx3-ubyte", testCount);
		this.testLabelLoader = new MatrixMnistLabelLoader("D:/测试数据/MNIST/t10k-labels.idx1-ubyte", testCount);
		// this.imageLoader = new
		// MatrixMnistImageLoader("F:/data/MNIST/train-images.idx3-ubyte",
		// trainCount);
		// this.labelLoader = new
		// MatrixMnistLabelLoader("F:/data/MNIST/train-labels.idx1-ubyte",
		// trainCount);
		// this.testImageLoader = new
		// MatrixMnistImageLoader("F:/data/MNIST/t10k-images.idx3-ubyte",
		// testCount);
		// this.testLabelLoader = new
		// MatrixMnistLabelLoader("F:/data/MNIST/t10k-labels.idx1-ubyte",
		// testCount);

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
			INDArray input = testImageInput.get(i);
			int rightLabel = MatrixMnistLabelLoader.getResult(testLabelList.get(i));
			int predict = MatrixMnistLabelLoader.getResult(network.predict(input));
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
		List<INDArray> imageInput = imageLoader.load();
		List<INDArray> label = labelLoader.load();
		System.out.println("开始训练，当前时间:" + TimeUtil.getNowTime());
		while (true) {
			epoch++;
			network.train(label, imageInput, 0.01, 1);
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
		MatrixMnistMain main = new MatrixMnistMain(60000, 10000);
		main.trainAndEvaluate();
		System.out.println(TimeUtil.getNowTime() + "训练结束,花费时间(秒):" + (System.currentTimeMillis() - begin) / 1000l);
	}
}
