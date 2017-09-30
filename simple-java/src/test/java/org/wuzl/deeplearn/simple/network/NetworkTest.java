package org.wuzl.deeplearn.simple.network;

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.Lists;

public class NetworkTest {
	public static void main(String[] args) {
		Network network = new Network(Lists.newArrayList(2, 2, 2));
		List<List<Double>> inputs = new ArrayList<>();
		inputs.add(Lists.newArrayList(0.1, 0.1));
		inputs.add(Lists.newArrayList(0.2, 0.2));
		inputs.add(Lists.newArrayList(0.3, 0.3));
		inputs.add(Lists.newArrayList(0.4, 0.4));
		List<List<Double>> labels = new ArrayList<>();
		labels.add(Lists.newArrayList(0.2, 0.2));
		labels.add(Lists.newArrayList(0.3, 0.3));
		labels.add(Lists.newArrayList(0.4, 0.5));
		labels.add(Lists.newArrayList(0.4, 0.5));
		network.train(inputs, labels, 0.3, 1);
		System.out.println(network.predict(Lists.newArrayList(0.5, 0.5)));
//		StringBuilder sb = new StringBuilder();
//		for (List<Double> input : inputs) {
//			sb.append("[");
//			for (int i = 0; i < input.size(); i++) {
//				sb.append(input.get(i));
//				if (i != input.size() - 1) {
//					sb.append(",");
//				}
//			}
//			sb.append("],");
//		}
//		System.out.println(sb);
//		sb = new StringBuilder();
//		for (List<Double> label : labels) {
//			sb.append("[");
//			for (int i = 0; i < label.size(); i++) {
//				sb.append(label.get(i));
//				if (i != label.size() - 1) {
//					sb.append(",");
//				}
//			}
//			sb.append("],");
//		}
//		System.out.println(sb);
	}
}
