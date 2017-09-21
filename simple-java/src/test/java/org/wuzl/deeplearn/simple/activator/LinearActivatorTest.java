package org.wuzl.deeplearn.simple.activator;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.wuzl.deeplearn.simple.Perceptron;

public class LinearActivatorTest {
	private Perceptron perceptron = new Perceptron(new LinearActivator(), 1);

	@Test
	public void shouldTestAndActivator() {
		this.train();
		// 验证
		System.out.println(perceptron);
		System.out.println("Work 3.4 years, monthly salary =" + perceptron.predict(new Double[] { 3.4 }));
		System.out.println("Work 3.4 years, monthly salary =" + perceptron.predict(new Double[] { 15.0 }));
		System.out.println("Work 3.4 years, monthly salary =" + perceptron.predict(new Double[] { 1.5 }));
		System.out.println("Work 3.4 years, monthly salary =" + perceptron.predict(new Double[] { 6.3 }));
	}

	private void train() {
		List<Double[]> inputVecArrayList = new ArrayList<Double[]>();
		// 输入向量列表
		inputVecArrayList.add(new Double[] { 5.0 });
		inputVecArrayList.add(new Double[] { 3.0 });
		inputVecArrayList.add(new Double[] { 8.0 });
		inputVecArrayList.add(new Double[] { 1.4 });
		inputVecArrayList.add(new Double[] { 10.1 });
		// 期望的输出列表，注意要与输入一一对应
		List<Double> labelList = new ArrayList<Double>();
		labelList.add(5500.0);
		labelList.add(2300.0);
		labelList.add(7600.0);
		labelList.add(1800.0);
		labelList.add(11400.0);
		// 循环 迭代10轮 学习速率为0.01
		perceptron.train(inputVecArrayList, labelList, 10, 0.01);
	}
}
