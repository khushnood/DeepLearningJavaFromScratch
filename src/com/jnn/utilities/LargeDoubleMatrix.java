package com.jnn.utilities;

import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;

import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
/**
 * 
 * @author Sorry I forgot the main author of this code. I have found hint on somewhere on stack overflow. 
 * Anyways this java code is for creating and manuplating a large matrix on hard disk. If you have less main memory this will work. 
 * It uses faster Input operations.
 *
 */

public class LargeDoubleMatrix implements Closeable {
	private static final int MAPPING_SIZE = 1 << 30;
	private final RandomAccessFile raf;
	private final int columns;
	private final int rows;
	private final List<MappedByteBuffer> mappings = new ArrayList<MappedByteBuffer>();
 
	/**
	 * 
	 * @param filename
	 * @param height
	 * @param width
	 * @throws IOException
	 */
	public LargeDoubleMatrix(String filename, int height, int width) throws IOException {
		this.raf = new RandomAccessFile(filename, "rw");
		try {
			this.columns = width;
			this.rows = height;
			long size = 8L * width * height;
			for (long offset = 0; offset < size; offset += MAPPING_SIZE) {
				long size2 = Math.min(size - offset, MAPPING_SIZE);
				mappings.add(raf.getChannel().map(FileChannel.MapMode.READ_WRITE, offset, size2));
			}
		} catch (IOException e) {
			raf.close();
			throw e;
		}
	}
	/**
	 * 
	 * @param filename
	 * @param height
	 * @param width
	 * @param isNew
	 * @throws IOException
	 */
	public LargeDoubleMatrix(String filename, int height, int width,boolean isNew) throws IOException {
		if(isNew) {
			File toDeletefile=new File(filename);
			toDeletefile.delete();
		}
		this.raf = new RandomAccessFile(filename, "rw");
		try {
			this.columns = width;
			this.rows = height;
			long size = 8L * width * height;
			for (long offset = 0; offset < size; offset += MAPPING_SIZE) {
				long size2 = Math.min(size - offset, MAPPING_SIZE);
				mappings.add(raf.getChannel().map(FileChannel.MapMode.READ_WRITE, offset, size2));
			}
		} catch (IOException e) {
			raf.close();
			throw e;
		}
	}
	/**
	 * 
	 * @param file
	 * @param height
	 * @param width
	 * @param isNew
	 * @throws IOException
	 */
	public LargeDoubleMatrix(File file, int height, int width,boolean isNew) throws IOException {
		if(isNew) {
						file.delete();
		}
		this.raf = new RandomAccessFile(file, "rw");
		try {
			this.columns = width;
			this.rows = height;
			long size = 8L * width * height;
			for (long offset = 0; offset < size; offset += MAPPING_SIZE) {
				long size2 = Math.min(size - offset, MAPPING_SIZE);
				mappings.add(raf.getChannel().map(FileChannel.MapMode.READ_WRITE, offset, size2));
			}
		} catch (IOException e) {
			raf.close();
			throw e;
		}
	}
	/**
	 * 
	 * @param height
	 * @param width
	 * @throws IOException
	 */
	public LargeDoubleMatrix(int height, int width) throws IOException {
		String tempPath = System.getProperty("java.io.tmpdir");
		File tmpFilePath = File.createTempFile("tmpMatFile", ".txt", new File(tempPath));
		this.raf = new RandomAccessFile(tmpFilePath, "rw");
		try {
			this.columns = width;
			this.rows = height;
			long size = 8L * width * height;
			for (long offset = 0; offset < size; offset += MAPPING_SIZE) {
				long size2 = Math.min(size - offset, MAPPING_SIZE);
				mappings.add(raf.getChannel().map(FileChannel.MapMode.READ_WRITE, offset, size2));
			}
		} catch (IOException e) {
			raf.close();
			throw e;
		}
	}
	/**
	 * 
	 * @param filePathToRead
	 * @param fromNodeColIndx
	 * @param toNodeColIndx
	 * @param edgeWeigthInd
	 * @param randomTime
	 * @return
	 * @throws IOException
	 */
	public static LargeDoubleMatrix creatMatrixFromFile(String filePathToRead,int fromNodeColIndx,int toNodeColIndx,int edgeWeigthInd,int randomTime ) throws IOException{
		String tempPath = System.getProperty("java.io.tmpdir");
		Scanner sc = null;
		FileInputStream inputStream = null;
		int max=-9999;
		int min=+9999999;
		 LargeDoubleMatrix matrixInFile =null;
        try {
        	
        	inputStream = new FileInputStream(filePathToRead);
			File tmpFilePath = File.createTempFile("tmpMatFile", ".txt", new File(tempPath));
			sc = new Scanner(inputStream, "UTF-8");
			//	n=10000;
				 
				while (sc.hasNextLine()) {
					String line = sc.nextLine();
					String[] oneRow = line.split("\t");
					int fromnodeId = Integer.parseInt(oneRow[fromNodeColIndx]);
					int tonodeId = Integer.parseInt(oneRow[toNodeColIndx]);
					int edgeWeight=Integer.parseInt(oneRow[edgeWeigthInd]);
					if(edgeWeight<=randomTime){
					max = Math.max(max, fromnodeId);
					max = Math.max(max, tonodeId);
					min = Math.min(min, fromnodeId);
					min = Math.min(min, tonodeId);
					}
				
				}
				int adjMatSize=(max+1);
				 matrixInFile = new LargeDoubleMatrix(tmpFilePath, adjMatSize, adjMatSize,true);
				 sc.close();
				 inputStream.close();
				 sc=null;
				 inputStream=null;
				
				 inputStream = new FileInputStream(filePathToRead);
				 sc = new Scanner(inputStream, "UTF-8");
				 
				 while (sc.hasNextLine()) {
						String line = sc.nextLine();
						String[] oneRow = line.split("\t");
						int fromnodeId = Integer.parseInt(oneRow[fromNodeColIndx]);
						int tonodeId = Integer.parseInt(oneRow[toNodeColIndx]);
						int edgeWeight=Integer.parseInt(oneRow[edgeWeigthInd]);
						if(edgeWeight<=randomTime){
							matrixInFile.set(fromnodeId, tonodeId, 1.0);
						}
						
					}
				sc.close();
				inputStream.close();
				return matrixInFile;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			throw e;
		}
	}
	/**
	 * 
	 * @param x
	 * @param y
	 * @return
	 */
	protected long position(int x, int y) {
		return (long) y * columns + x;
	}

	public int width() {
		return columns;
	}

	public int height() {
		return rows;
	}
	/**
	 * 
	 * @param y
	 * @param x
	 * @return
	 */

	public double get(int y, int x) {
		assert x >= 0 && x < columns;
		assert y >= 0 && y < rows;
		long p = position(x, y) * 8;
		int mapN = (int) (p / MAPPING_SIZE);
		int offN = (int) (p % MAPPING_SIZE);
		return mappings.get(mapN).getDouble(offN);
	}
/**
 * 
 * @param y
 * @param x
 * @param d
 */
	public void set(int y, int x, double d) {
		assert x >= 0 && x < columns;
		assert y >= 0 && y < rows;
		long p = position(x, y) * 8;
		int mapN = (int) (p / MAPPING_SIZE);
		int offN = (int) (p % MAPPING_SIZE);
		mappings.get(mapN).putDouble(offN, d);
	}

	public void close() throws IOException {
		for (MappedByteBuffer mapping : mappings)
			clean(mapping);
		raf.close();
	}

	private void clean(MappedByteBuffer mapping) {
		if (mapping == null)
			return;
		mapping.clear();
		/*
		 * Cleaner cleaner = ((DirectBuffer) mapping).cleaner(); if (cleaner != null)
		 * cleaner.clean();
		 */
	}

	 

	    /**
	     * {@inheritDoc}
	     */
	public double[] getColumn(int column) {
	        double[] values = new double[rows];
	        for (int row = 0; row < rows; ++row)
	            values[row] = get(row, column);
	        return values;
	    }
	
	public double[] getRow(int row) {
		
		double[] values = new double[columns];
        for (int col = 0; col < columns; ++col)
            values[col] = get(row, col);
        return values;
		
	}
	

    
	
	private static long usedMemory() {
		return Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
	}
/**
 * create and test your large matrix operation from your hard disk
 * @param args
 */
	public static void main(String[] args) {

		try {
			long start = System.nanoTime();
			final long used0 = usedMemory();
			LargeDoubleMatrix matrix = new LargeDoubleMatrix("\\tmpMat.txt", 38549, 38549);
			for (int i = 0; i < matrix.width(); i++)
				matrix.set(i, i, 1);
			for (int i = 0; i < matrix.width(); i++) {
				if (matrix.get(i, i) == 1) {
					// System.out.println("diagonal element");
				}
			}			
			long time = System.nanoTime() - start;
			final long used = usedMemory() - used0;
			if (used == 0)
				System.err.println("You need to use -XX:-UsedTLAB to see small changes in memory usage.");
			System.out.printf("Setting the diagonal took %,d ms, Heap used is %,d KB%n", time / 1000 / 1000,
					used / 1024);
			matrix.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}