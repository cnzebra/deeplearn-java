package org.wuzl.deeplearn.simple.network;

import java.util.List;

import com.google.common.collect.Lists;

public class NetworkGradientCheck {
	public double calcNetworkError(List<Double> vec1, List<Double> vec2) {
		double result = 0.0;
		for (int i = 0; i < vec1.size(); i++) {
			result += Math.pow(vec1.get(i) - vec2.get(i), 2) / 2;
		}
		return result;
	}

	/**
	 * 检查梯度
	 * 
	 * @param network
	 * @param simpleInput
	 *            简单输入
	 * @param simpleLabel
	 *            简单输出
	 */
	public void gradientCheck(Network network, List<Double> simpleInput, List<Double> simpleLabel) {
		// 获取网络在当前样本下每个连接的梯度
		network.calcGradient(simpleInput, simpleLabel);
		// 对每个权重做梯度检查
		for (NetwordConnection connection : network.getConnectionList()) {
			// 获取指定连接的梯度
			double actualGradient = connection.getGradient();
			// 增加一个很小的值，计算网络的误差
			double epsilon = 0.0001;
			connection.setWeight(connection.getWeight() + epsilon);
			double error1 = calcNetworkError(network.predict(simpleInput), simpleLabel);
			// 减去一个很小的值，计算网络的误差
			connection.setWeight(connection.getWeight() - 2 * epsilon);
			double error2 = calcNetworkError(network.predict(simpleInput), simpleLabel);
			// 根据式6计算期望的梯度值
			double expectedGradient = (error2 - error1) / (2 * epsilon);
			System.out
					.println("expected gradient: \t" + expectedGradient + "\nactual gradient: \t" + actualGradient);
		}
	}

	public static void main(String[] args) {
		// System.out.println(new
		// NetworkGradientCheck().calcNetworkError(Lists.newArrayList(1.0, 2.0,3.0),
		// Lists.newArrayList(10.0, 20.0,30.0)));
		Network network = new Network(Lists.newArrayList(2, 2, 2));
		new NetworkGradientCheck().gradientCheck(network, Lists.newArrayList(0.9, 0.1), Lists.newArrayList(0.9, 0.1));
	}
}
