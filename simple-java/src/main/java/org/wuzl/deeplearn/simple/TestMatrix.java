package org.wuzl.deeplearn.simple;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.Pow;
import org.nd4j.linalg.factory.Nd4j;

public class TestMatrix {
	public static void main(String[] args) {

		INDArray arr1 = Nd4j.ones(2, 2);
		arr1.add(1);// 不改变原值生成新对象
		System.out.println(arr1);
		arr1.addi(1);// 改变原值
		System.out.println(arr1);
		System.out.println(arr1.add(1));
		System.out.println(arr1);
		INDArray exp = Nd4j.getExecutioner().execAndReturn(new Exp(arr1));
		System.out.println(exp);
		System.out.println(arr1);

		exp = Nd4j.getExecutioner().execAndReturn(new Exp(arr1.dup()));
		System.out.println(exp);
		System.out.println(arr1);
		// 测试算法
		System.out.println(">>>>>>>> ");
		INDArray ndArray = Nd4j.ones(2, 2);
		System.out.println(ndArray.length());
		// 1.0 / (1.0 + np.exp(-weighted_input))
		System.out.println(ndArray);
		INDArray expInd = Nd4j.getExecutioner().execAndReturn(new Exp(ndArray.muli(-1)));
		expInd.addi(1.0);
		// expInd = Nd4j.ones(2, 2).divi(expInd);//用除法
		// Nd4j.getExecutioner().execAndReturn(new Pow(expInd, -1));//用pow
		expInd.rdivi(1);
		System.out.println(expInd);
		System.out.println(">>>>>>>>>");
		System.out.println(Nd4j.zeros(3, 1).length());

		ndArray = Nd4j.ones(2, 2);
		ndArray.addi(5);
		System.out.println(ndArray);
		System.out.println(ndArray.sub(1));
		System.out.println(ndArray.rsubi(1));
		System.out.println(ndArray);
		INDArray weightArray = Nd4j.ones(2, 2);
		weightArray.addi(5);
		INDArray input = Nd4j.ones(2, 2);
		input.addi(3);
		System.out.println(weightArray.mmul(input));
		System.out.println(weightArray);
		input = Nd4j.ones(3, 2);
		input.addi(3);
		System.out.println(input);
		System.out.println(input.transpose());
		input = Nd4j.ones(5, 3);
		System.out.println(">>>>>");
		System.out.println(input);
		input.putScalar(3, 2, 5);
		System.out.println(input);
		System.out.println(input.getDouble(3,2));
		input.putScalar(3,1, input.getDouble(3,1)+5);
		System.out.println(input);
	}
}
