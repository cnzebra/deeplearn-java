package org.wuzl.deeplearn.simple.util;

import org.junit.Test;

public class ByteUtilTest {
	@Test
	public void shouldGetUnsigned() {
		byte b=(byte)234; 
		System.out.println(b);
		System.out.println(ByteUtil.getUnsigned(b));
	}
	@Test
	public void shouldGetUnsignedDouble() {
		byte b=(byte)234; 
		System.out.println(b);
		System.out.println(ByteUtil.getUnsignedDouble(b));
	}
}
