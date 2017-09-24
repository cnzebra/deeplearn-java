package org.wuzl.deeplearn.simple.util;

public class ByteUtil {
	public static short getUnsigned(byte b) {
		short s = (short) 0xff;
		s = (short) (s & b);
		return s;
	}
	
	public static double getUnsignedDouble(byte b) {
		return getUnsigned(b)+0.0;
	}
	
}
