package org.wuzl.deeplearn.simple.util;

import java.util.Random;

public class MathUtil {
	public static double random() {
		return new Random().nextInt(10) / 100.0 * (Math.random() > 0.5 ? 1 : -1);
	}

	public static double sigmoid(double input) {
		return 1.0 / (1 + Math.exp(-input));
	}

	public static void main(String[] args) {
		while (true) {
			System.out.println(random());
			try {
				Thread.sleep(100l);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}
