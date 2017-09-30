package org.wuzl.deeplearn.simple;

import org.nd4j.linalg.factory.Nd4j;

public class TestRandomMatrix {
	public static void main(String[] args) {
		System.out.println(Nd4j.randn(3, 2));

		System.out.println(Nd4j.rand(3, 2));// 0-1
		System.out.println(Nd4j.rand(3, 2, 0,256, Nd4j.getRandom()));
	}
}
