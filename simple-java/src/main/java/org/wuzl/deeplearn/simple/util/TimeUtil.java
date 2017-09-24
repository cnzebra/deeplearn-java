package org.wuzl.deeplearn.simple.util;

import java.text.SimpleDateFormat;
import java.util.Date;

public class TimeUtil {
	public static String getNowTime() {
		return new SimpleDateFormat("yyyy年MM月dd日 HH时:mm分:ss秒").format(new Date());
	}
}
