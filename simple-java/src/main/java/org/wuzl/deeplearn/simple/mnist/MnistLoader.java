package org.wuzl.deeplearn.simple.mnist;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * 数据加载器基类
 * 
 * @author Administrator
 *
 */
public class MnistLoader {
	// 数据文件路径
	private final String path;
	// 文件中的样本个数
	private final int count;

	protected final ByteBuffer buffer;

	public MnistLoader(String path, int count) {
		this.path = path;
		this.count = count;
		this.buffer = readFileData();
	}

	ByteBuffer readFileData() {
		File file = new File(path);
		if (!file.exists()) {
			throw new RuntimeException("指定文件不存在" + path);
		}
		InputStream is = null;
		try {
			is = new FileInputStream(file);
			byte[] bytes = new byte[is.available()];
			is.read(bytes);
			ByteBuffer buffer = ByteBuffer.wrap(bytes);
			return buffer;
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (is != null) {
				try {
					is.close();
				} catch (IOException e) {
				}
			}
		}
		return null;
	}

	public int getCount() {
		return count;
	}

	public static void main(String[] args) {
		// MnistLoader loader = new MnistLoader("F:/data/MNIST/train-images.idx3-ubyte",
		// 100);
		// System.out.println(loader.readFileData().length / 1024 / 1024);
	}
}
