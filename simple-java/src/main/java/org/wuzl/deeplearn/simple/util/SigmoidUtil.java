package org.wuzl.deeplearn.simple.util;

public class SigmoidUtil {
	public static double sigmoid(double input) {
		return 1.0 / (1 + Math.exp(-input));
	}

	public static void main(String[] args) {
		System.out.println(sigmoid(1.1));
	}
}
