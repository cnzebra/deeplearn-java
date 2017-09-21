package org.wuzl.deeplearn.simple.activator;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.wuzl.deeplearn.simple.Perceptron;

/**
 * 测试线性
 * 
 * @author ziliang.wu
 *
 */
public class AndActivatorTest {
	private Perceptron perceptron = new Perceptron(new AndActivator(), 2);

	@Test
	public void shouldTestAndActivator() {
		this.train();
		// 验证
		System.out.println(perceptron);
		System.out.println("1 and 1 =" + perceptron.predict(new Double[] { 1.0, 1.0 }));
		System.out.println("0 and 0 = " + perceptron.predict(new Double[] { 0.0, 0.0 }));
		System.out.println("1 and 0 =" + perceptron.predict(new Double[] { 1.0, 0.0 }));
		System.out.println("0 and 1 =" + perceptron.predict(new Double[] { 0.0, 1.0 }));
	}

	private void train() {
		List<Double[]> inputVecArrayList = new ArrayList<Double[]>();
		// 输入向量列表
		inputVecArrayList.add(new Double[] { 1.0, 1.0 });
		inputVecArrayList.add(new Double[] { 0.0, 0.0 });
		inputVecArrayList.add(new Double[] { 1.0, 0.0 });
		inputVecArrayList.add(new Double[] { 0.0, 1.0 });
		// 期望的输出列表，注意要与输入一一对应
		List<Double> labelList = new ArrayList<Double>();
		labelList.add(1.0);
		labelList.add(0.0);
		labelList.add(0.0);
		labelList.add(0.0);
		// 循环 迭代10轮 实际4轮就可以, 学习速率为0.1
		perceptron.train(inputVecArrayList, labelList, 10, 0.1);
	}
}
