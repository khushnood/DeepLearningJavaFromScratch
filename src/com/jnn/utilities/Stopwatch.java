package com.jnn.utilities;

/**
 * 
 * Credit: http://introcs.cs.princeton.edu/java/32class/Stopwatch.java.html
 * 
 */

public class Stopwatch {
	private final long start;

	public Stopwatch() { 
		start = System.currentTimeMillis();
	}

	// return time (in seconds) since this object was created
	public double elapsedTime() {
		long now = System.currentTimeMillis();
		return (now - start) / 1000.0;
	}
	
}