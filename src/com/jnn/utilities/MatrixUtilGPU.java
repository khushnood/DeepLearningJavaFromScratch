package com.jnn.utilities;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandDestroyGenerator;
import static jcuda.jcurand.JCurand.curandGenerateUniform;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.JCuda;
import jcuda.vec.VecDouble;
import jcuda.vec.VecFloat;

import com.jnn.consts.IPredictionModelConstants;
import com.jnn.enums.DeepNNActivations;
import com.jnn.enums.DeepNNKeysForMaps;
import com.jnn.enums.ProbablityDistributionTypes;
import com.jnn.enums.SimpleMathOperations;
import com.jnn.enums.StandardMathFunctions;
import com.jnn.utilities.LargeDoubleMatrix;


/**
 * Copyright 2020
 * 
 * @author Khushnood Abbas
 * @email khushnood.abbas@gmail.com
 * 
 *        Unless required by applicable law or agreed to in writing, software
 *        distributed under the License is distributed on an "AS IS" BASIS,
 *        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 *        implied. See the License for the specific language governing
 *        permissions and limitations under the License.
 *
 */

public class MatrixUtilGPU {

	private static Random random;
	private static long randomSeed;

	static {
		randomSeed = System.currentTimeMillis();
		random = new Random(randomSeed);
	}
/**
 * 
 * @param s
 */
	public static void setSeed(long s) {
		randomSeed = s;
		random = new Random(randomSeed);
	}

	public static long getSeed() {
		return randomSeed;
	}

	public static double uniform() {
		return random.nextDouble();
	}

	public static double normal() {
		return random.nextDouble();
	}
	/**
	 * 
	 * @param n
	 * @return
	 */

	public static int uniform(int n) {
		if (n <= 0) {
			throw new IllegalArgumentException("argument must be positive: "
					+ n);
		}
		return random.nextInt(n);
	}
/**
 * 
 * @param n
 * @return
 */
	public static long uniform(long n) {
		if (n <= 0L) {
			throw new IllegalArgumentException("argument must be positive: "
					+ n);
		}

		long r = random.nextLong();
		long m = n - 1;

		if ((n & m) == 0L) {
			return r & m;
		}

		long u = r >>> 1;
		while (u + m - (r = u % n) < 0L) {
			u = random.nextLong() >>> 1;
		}
		return r;
	}

	 /**
	  * 
	  * @param a
	  * @param b
	  * @return
	  */
	public static int uniform(int a, int b) {
		if ((b <= a) || ((long) b - a >= Integer.MAX_VALUE)) {
			throw new IllegalArgumentException("invalid range: [" + a + ", "
					+ b + ")");
		}
		return a + uniform(b - a);
	}

	/**
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static double uniform(double a, double b) {
		if (!(a < b)) {
			throw new IllegalArgumentException("invalid range: [" + a + ", "
					+ b + ")");
		}
		return a + uniform() * (b - a);
	}
/**
 * 
 * @param a
 * @param b
 * @return
 */
	public static double normal(double a, double b) {
		if (!(a < b)) {
			throw new IllegalArgumentException("invalid range: [" + a + ", "
					+ b + ")");
		}
		return normal();
	}

	/**
	 * 
	 * @param m
	 * @param n
	 * @return
	 */
	public static double[][] random(int m, int n) {
		double[][] a = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				a[i][j] = uniform(0.0, 1.0);
			}
		}
		return a;
	}
/**
 * 
 * @param m
 * @param n
 * @param probType
 * @return
 */
	public static double[][] random(int m, int n,
			ProbablityDistributionTypes probType) {
		double[][] a = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (probType == ProbablityDistributionTypes.UNIFORM) {
					a[i][j] = uniform(0.0, 1.0);
				} else if (probType == ProbablityDistributionTypes.NORMAL) {
					a[i][j] = normal();
				}

			}
		}
		return a;
	}
/**
 * 
 * @param m
 * @param n
 * @return
 */
	public static double[][] zeros(int m, int n) {
		double[][] a = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				a[i][j] = 0.0;
			}
		}
		return a;
	}

	/**
	 * Transpose of a matrix
	 *
	 * @param a
	 *            matrix
	 * @return b = A^T
	 */
	public static double[][] T(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] b = new double[n][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				b[j][i] = a[i][j];
			}
		}
		return b;
	}
/**
 * get transpose of a matrix
 * @param a
 * @return
 */
	public static int[][] T(int[][] a) {
		int m = a.length;
		int n = a[0].length;
		int[][] b = new int[n][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				b[j][i] = a[i][j];
			}
		}
		return b;
	}
/**
 * 
 * @param a
 * @return
 */
	public static float[][] T(float[][] a) {
		int m = a.length;
		int n = a[0].length;
		float[][] b = new float[n][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				b[j][i] = a[i][j];
			}
		}
		return b;
	}

	/**
	 * get log of all the elements of a matrix
	 * 
	 * @param a
	 * @return
	 */
	public static double[][] log(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] b = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				b[i][j] = Math.log(a[i][j]);
			}
		}
		return b;
	}
/**
 * Get squar root of matrix elements
 * @param a
 * @return
 */
	public static double[][] sqrt(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] b = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				b[i][j] = Math.sqrt(a[i][j]);
			}
		}
		return b;
	}

	/**
	 * @param a
	 *            matrix
	 * @param b
	 *            matrix
	 * @return c = a + b
	 * @throws Exception
	 */
	public static double[][] add(double[][] a, double[][] b) throws Exception {

		return simplePointWiseOperations(a, b, SimpleMathOperations.ADD);
	}

	public static float[][] add(float[][] a, float[][] b) throws Exception {

		return simplePointWiseOperations(a, b, SimpleMathOperations.ADD);
	}

	/**
	 * khushnood Abbas
	 * 
	 * @param a
	 * @param b
	 * @param operationType
	 * @return
	 * @throws Exception
	 */
	public static double[][] simplePointWiseOperations(double[][] a,
			double[][] b, SimpleMathOperations operationType) throws Exception {

		int m = a.length;
		int n = a[0].length;

		GpuUtil.initGPUForVectorOpt();
		VecDouble.init();

		CUdeviceptr tmpPonitera = null;
		CUdeviceptr tmpPoniterb = null;
		CUdeviceptr resultPoniter = new CUdeviceptr();

		if (a.length != m || b[0].length != n) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		int sigleArraySize = m * n;
		double[] y = new double[sigleArraySize];

		cuMemAlloc(resultPoniter, (sigleArraySize) * Sizeof.DOUBLE);

		tmpPonitera = new CUdeviceptr();
		cuMemAlloc(tmpPonitera, (sigleArraySize) * Sizeof.DOUBLE);
		tmpPoniterb = new CUdeviceptr();
		cuMemAlloc(tmpPoniterb, (sigleArraySize) * Sizeof.DOUBLE);

		double[] tmpa = GpuUtil.matrixtoArray(a, m, n);
		double[] tmpb = GpuUtil.matrixtoArray(b, m, n);

		cuMemcpyHtoD(tmpPonitera, Pointer.to(tmpa), sigleArraySize
				* Sizeof.DOUBLE);
		cuMemcpyHtoD(tmpPoniterb, Pointer.to(tmpb), sigleArraySize
				* Sizeof.DOUBLE);

		if (operationType == SimpleMathOperations.ADD) {
			VecDouble.add(sigleArraySize, resultPoniter, tmpPonitera,
					tmpPoniterb);
		} else if (operationType == SimpleMathOperations.SUB) {
			VecDouble.sub(sigleArraySize, resultPoniter, tmpPonitera,
					tmpPoniterb);
		} else if (operationType == SimpleMathOperations.MUL) {
			VecDouble.mul(sigleArraySize, resultPoniter, tmpPonitera,
					tmpPoniterb);
		} else if (operationType == SimpleMathOperations.DIV) {
			VecDouble.div(sigleArraySize, resultPoniter, tmpPonitera,
					tmpPoniterb);
		} else {
			throw new Exception("un supported operations: " + operationType);
		}

		cuMemcpyDtoH(Pointer.to(y), resultPoniter, (sigleArraySize)
				* Sizeof.DOUBLE);

		tmpa = null;
		tmpb = null;
		double[][] resultMatr = GpuUtil.arrayToDoubleMatrix(y, m, n);
		y = null;
		GpuUtil.freeMemory(resultPoniter);
		GpuUtil.freeMemory(tmpPonitera, tmpPoniterb);
		VecDouble.shutdown();
		GpuUtil.shutDownGpuForVectorOpt();

		return resultMatr;

	}
/**
 * For point wise matrix operations
 * @param a
 * @param b
 * @param operationType
 * @return
 * @throws Exception
 */
	public static float[][] simplePointWiseOperations(float[][] a, float[][] b,
			SimpleMathOperations operationType) throws Exception {

		int m = a.length;
		int n = a[0].length;

		GpuUtil.initGPUForVectorOpt();
		VecFloat.init();
		CUdeviceptr tmpPonitera = new CUdeviceptr();
		CUdeviceptr resultPoniter = new CUdeviceptr();

		if (a.length != m || b[0].length != n) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		int sigleArraySize = m * n;
		float[] y = new float[sigleArraySize];
		CUdeviceptr tmpPoniterb = null;

		cuMemAlloc(resultPoniter, (sigleArraySize) * Sizeof.FLOAT);

		tmpPonitera = new CUdeviceptr();
		cuMemAlloc(tmpPonitera, (sigleArraySize) * Sizeof.FLOAT);
		tmpPoniterb = new CUdeviceptr();
		cuMemAlloc(tmpPoniterb, (sigleArraySize) * Sizeof.FLOAT);

		float[] tmpa = GpuUtil.matrixtoArray(a, m, n);
		float[] tmpb = GpuUtil.matrixtoArray(b, m, n);

		cuMemcpyHtoD(tmpPonitera, Pointer.to(tmpa), sigleArraySize
				* Sizeof.FLOAT);
		cuMemcpyHtoD(tmpPoniterb, Pointer.to(tmpb), sigleArraySize
				* Sizeof.FLOAT);

		if (operationType == SimpleMathOperations.ADD) {
			VecFloat.add(sigleArraySize, resultPoniter, tmpPonitera,
					tmpPoniterb);
		} else if (operationType == SimpleMathOperations.SUB) {
			VecFloat.sub(sigleArraySize, resultPoniter, tmpPonitera,
					tmpPoniterb);
		} else if (operationType == SimpleMathOperations.MUL) {
			VecFloat.mul(sigleArraySize, resultPoniter, tmpPonitera,
					tmpPoniterb);
		} else if (operationType == SimpleMathOperations.DIV) {
			VecFloat.div(sigleArraySize, resultPoniter, tmpPonitera,
					tmpPoniterb);
		} else {
			throw new Exception("un supported operations: " + operationType);
		}

		cuMemcpyDtoH(Pointer.to(y), resultPoniter, (sigleArraySize)
				* Sizeof.FLOAT);

		tmpa = null;
		tmpb = null;
		float[][] resultMatr = GpuUtil.arrayToFloatMatrix(y, m, n);
		y = null;
		GpuUtil.freeMemory(tmpPonitera);
		GpuUtil.freeMemory(resultPoniter);
		VecFloat.shutdown();
		GpuUtil.shutDownGpuForVectorOpt();
		// System.out.print("+ ");
		return resultMatr;

	}

	/* This method is pointwise operations where second operand is a scalar. The scalar value is rescaled to matrix dimensions
	 * khushnood Abbas
	 * 
	 * @param a
	 * 
	 * @param b
	 * 
	 * @param operationType
	 * 
	 * @return
	 */
	public static double[][] simplePointWiseOperationsWithScalarAsSecondOperand(
			double[][] a, double b, SimpleMathOperations operationType)
			throws Exception {

		int m = a.length;
		int n = a[0].length;
		GpuUtil.initGPUForVectorOpt();
		VecDouble.init();
		CUdeviceptr tmpPonitera = new CUdeviceptr();
		;
		// CUdeviceptr tmpPoniterb=null;
		CUdeviceptr resultPoniter = new CUdeviceptr();
		int dataSize = m * n;

		double[] y = new double[dataSize];

		cuMemAlloc(resultPoniter, dataSize * Sizeof.DOUBLE);

		cuMemAlloc(tmpPonitera, dataSize * Sizeof.DOUBLE);
		double[] tmpa = GpuUtil.matrixtoArray(a, m, n);

		cuMemcpyHtoD(tmpPonitera, Pointer.to(tmpa), dataSize * Sizeof.DOUBLE);
		if (operationType == SimpleMathOperations.ADD) {
			VecDouble.addScalar(dataSize, resultPoniter, tmpPonitera, b);
		} else if (operationType == SimpleMathOperations.SUB) {
			VecDouble.subScalar(dataSize, resultPoniter, tmpPonitera, b);
		} else if (operationType == SimpleMathOperations.MUL) {
			VecDouble.mulScalar(dataSize, resultPoniter, tmpPonitera, b);
		} else if (operationType == SimpleMathOperations.DIV) {
			VecDouble.divScalar(dataSize, resultPoniter, tmpPonitera, b);
		} else {
			throw new Exception("un supported operations: " + operationType);
		}
		cuMemcpyDtoH(Pointer.to(y), resultPoniter, dataSize * Sizeof.DOUBLE);

		tmpa = null;

		double[][] resultMatr = GpuUtil.arrayToDoubleMatrix(y, m, n);
		y = null;
		GpuUtil.freeMemory(tmpPonitera);

		GpuUtil.freeMemory(resultPoniter);
		VecDouble.shutdown();
		GpuUtil.shutDownGpuForVectorOpt();
		// System.out.print("+ ");
		return resultMatr;

	}
/**
 * This method is pointwise operations where first operand is a scalar. The scalar value is rescaled to matrix dimensions
 * @param b
 * @param a
 * @param operationType
 * @return
 * @throws Exception
 */
	public static double[][] simplePointWiseOperationsWithScalarAsFirstOperand(
			double b, double[][] a, SimpleMathOperations operationType)
			throws Exception {

		int m = a.length;
		int n = a[0].length;
		GpuUtil.initGPUForVectorOpt();
		VecDouble.init();
		CUdeviceptr tmpPonitera = new CUdeviceptr();
		;
		// CUdeviceptr tmpPoniterb=null;
		CUdeviceptr resultPoniter = new CUdeviceptr();
		int dataSize = m * n;

		double[] y = new double[dataSize];

		cuMemAlloc(resultPoniter, dataSize * Sizeof.DOUBLE);

		cuMemAlloc(tmpPonitera, dataSize * Sizeof.DOUBLE);
		double[] tmpa = GpuUtil.matrixtoArray(a, m, n);

		cuMemcpyHtoD(tmpPonitera, Pointer.to(tmpa), dataSize * Sizeof.DOUBLE);
		if (operationType == SimpleMathOperations.ADD) {
			VecDouble.scalarAdd(dataSize, resultPoniter, b, tmpPonitera);
		} else if (operationType == SimpleMathOperations.SUB) {
			VecDouble.scalarSub(dataSize, resultPoniter, b, tmpPonitera);
		} else if (operationType == SimpleMathOperations.MUL) {
			VecDouble.scalarMul(dataSize, resultPoniter, b, tmpPonitera);
		} else if (operationType == SimpleMathOperations.DIV) {
			VecDouble.scalarDiv(dataSize, resultPoniter, b, tmpPonitera);
		} else {
			throw new Exception("un supported operations: " + operationType);
		}
		cuMemcpyDtoH(Pointer.to(y), resultPoniter, dataSize * Sizeof.DOUBLE);

		tmpa = null;

		double[][] resultMatr = GpuUtil.arrayToDoubleMatrix(y, m, n);
		y = null;
		GpuUtil.freeMemory(tmpPonitera);

		GpuUtil.freeMemory(resultPoniter);
		VecDouble.shutdown();
		GpuUtil.shutDownGpuForVectorOpt();
		// System.out.print("+ ");
		return resultMatr;

	}
/**
 *  This method is pointwise operations where first operand is a scalar. The scalar value is rescaled to matrix dimensions
 * @param b
 * @param a
 * @param operationType
 * @return
 * @throws Exception
 */

	public static float[][] simplePointWiseOperationsWithScalarAsFirstOperand(
			float b, float[][] a, SimpleMathOperations operationType)
			throws Exception {

		int m = a.length;
		int n = a[0].length;
		GpuUtil.initGPUForVectorOpt();
		VecFloat.init();
		CUdeviceptr tmpPonitera = new CUdeviceptr();
		;
		// CUdeviceptr tmpPoniterb=null;
		CUdeviceptr resultPoniter = new CUdeviceptr();
		int dataSize = m * n;

		float[] y = new float[dataSize];

		cuMemAlloc(resultPoniter, dataSize * Sizeof.FLOAT);

		cuMemAlloc(tmpPonitera, dataSize * Sizeof.FLOAT);
		float[] tmpa = GpuUtil.matrixtoArray(a, m, n);

		cuMemcpyHtoD(tmpPonitera, Pointer.to(tmpa), dataSize * Sizeof.FLOAT);
		if (operationType == SimpleMathOperations.ADD) {
			VecFloat.scalarAdd(dataSize, resultPoniter, b, tmpPonitera);
		} else if (operationType == SimpleMathOperations.SUB) {
			VecFloat.scalarSub(dataSize, resultPoniter, b, tmpPonitera);
		} else if (operationType == SimpleMathOperations.MUL) {
			VecFloat.scalarMul(dataSize, resultPoniter, b, tmpPonitera);
		} else if (operationType == SimpleMathOperations.DIV) {
			VecFloat.scalarDiv(dataSize, resultPoniter, b, tmpPonitera);
		} else {
			throw new Exception("un supported operations: " + operationType);
		}
		cuMemcpyDtoH(Pointer.to(y), resultPoniter, dataSize * Sizeof.FLOAT);

		tmpa = null;

		float[][] resultMatr = GpuUtil.arrayToFloatMatrix(y, m, n);
		y = null;
		GpuUtil.freeMemory(tmpPonitera);

		GpuUtil.freeMemory(resultPoniter);
		VecFloat.shutdown();
		GpuUtil.shutDownGpuForVectorOpt();
		// System.out.print("+ ");
		return resultMatr;

	}
	/**
	 * 
	 * @param a
	 * @param b
	 * @param operationType
	 * @return
	 * @throws Exception
	 */

	public static float[][] simplePointWiseOperationsWithScalarAsSecondOperand(
			float[][] a, float b, SimpleMathOperations operationType)
			throws Exception {

		int m = a.length;
		int n = a[0].length;
		GpuUtil.initGPUForVectorOpt();
		VecFloat.init();
		CUdeviceptr tmpPonitera = new CUdeviceptr();

		// CUdeviceptr tmpPoniterb=null;
		CUdeviceptr resultPoniter = new CUdeviceptr();
		int dataSize = m * n;

		float[] y = new float[dataSize];

		cuMemAlloc(resultPoniter, dataSize * Sizeof.FLOAT);

		cuMemAlloc(tmpPonitera, dataSize * Sizeof.FLOAT);
		float[] tmpa = GpuUtil.matrixtoArray(a, m, n);

		cuMemcpyHtoD(tmpPonitera, Pointer.to(tmpa), dataSize * Sizeof.FLOAT);
		if (operationType == SimpleMathOperations.ADD) {
			VecFloat.addScalar(dataSize, resultPoniter, tmpPonitera, b);
		} else if (operationType == SimpleMathOperations.SUB) {
			VecFloat.subScalar(dataSize, resultPoniter, tmpPonitera, b);
		} else if (operationType == SimpleMathOperations.MUL) {
			VecFloat.mulScalar(dataSize, resultPoniter, tmpPonitera, b);
		} else if (operationType == SimpleMathOperations.DIV) {
			VecFloat.divScalar(dataSize, resultPoniter, tmpPonitera, b);
		} else {
			throw new Exception("un supported operations: " + operationType);
		}
		cuMemcpyDtoH(Pointer.to(y), resultPoniter, dataSize * Sizeof.FLOAT);

		tmpa = null;

		float[][] resultMatr = GpuUtil.arrayToFloatMatrix(y, m, n);
		y = null;
		GpuUtil.freeMemory(tmpPonitera);

		GpuUtil.freeMemory(resultPoniter);
		VecFloat.shutdown();
		GpuUtil.shutDownGpuForVectorOpt();
		// System.out.print("+ ");
		return resultMatr;

	}
/**
 * This method applies elementwise functions such as log, sqrt etc
 * @param a
 * @param activationType
 * @return
 * @throws Exception
 */
	public static double[][] simplePointWiseFunctions(double[][] a,
			StandardMathFunctions activationType) throws Exception {

		int m = a.length;
		int n = a[0].length;
		GpuUtil.initGPUForVectorOpt();
		VecDouble.init();
		CUdeviceptr tmpPonitera = new CUdeviceptr();
		;
		// CUdeviceptr tmpPoniterb=null;
		CUdeviceptr resultPoniter = new CUdeviceptr();
		int dataSize = m * n;
		double[] y = new double[dataSize];

		cuMemAlloc(resultPoniter, dataSize * Sizeof.DOUBLE);
		cuMemAlloc(tmpPonitera, dataSize * Sizeof.DOUBLE);
		double[] tmpa = GpuUtil.matrixtoArray(a, m, n);

		cuMemcpyHtoD(tmpPonitera, Pointer.to(tmpa), dataSize * Sizeof.DOUBLE);

		if (activationType == StandardMathFunctions.EXP) {
			VecDouble.exp(dataSize, resultPoniter, tmpPonitera);
		} else if (activationType == StandardMathFunctions.LOG) {
			VecDouble.log(dataSize, resultPoniter, tmpPonitera);
		} else if (activationType == StandardMathFunctions.SQRT) {
			VecDouble.sqrt(dataSize, resultPoniter, tmpPonitera);
		} else {
			throw new Exception("un supported operations: " + activationType);
		}
		cuMemcpyDtoH(Pointer.to(y), resultPoniter, dataSize * Sizeof.DOUBLE);
		double[][] resultMat = GpuUtil.arrayToDoubleMatrix(y, m, n);
		y = null;
		tmpa = null;
		;

		GpuUtil.freeMemory(tmpPonitera, resultPoniter);
		VecDouble.shutdown();
		GpuUtil.shutDownGpuForVectorOpt();
		return resultMat;

	}
/**
 * Different kinds of nonlinear activation functions. This is not complete as jCuda dont provide all functions.
 * @param a
 * @param activationType
 * @return
 * @throws Exception
 */
	public static double[][] activations(double[][] a,
			DeepNNActivations activationType) throws Exception {

		int m = a.length;
		int n = a[0].length;
		GpuUtil.initGPUForVectorOpt();
		VecDouble.init();
		CUdeviceptr tmpPonitera = new CUdeviceptr();
		;
		// CUdeviceptr tmpPoniterb=null;
		CUdeviceptr resultPoniter = new CUdeviceptr();
		int dataSize = m * n;
		double[] y = new double[dataSize];

		cuMemAlloc(resultPoniter, dataSize * Sizeof.DOUBLE);
		cuMemAlloc(tmpPonitera, dataSize * Sizeof.DOUBLE);
		double[] tmpa = GpuUtil.matrixtoArray(a, m, n);

		cuMemcpyHtoD(tmpPonitera, Pointer.to(tmpa), dataSize * Sizeof.DOUBLE);

		if (activationType == DeepNNActivations.TANH) {
			VecDouble.tanh(dataSize, resultPoniter, tmpPonitera);
		} else if (activationType == DeepNNActivations.SIGMOID) {
			// tmpPoniterb=GpuUtil.getGpuMemoryPointer(a[j], n);

			VecDouble.mulScalar(dataSize, tmpPonitera, tmpPonitera, -1.0);
			VecDouble.exp(dataSize, tmpPonitera, tmpPonitera);
			VecDouble.addScalar(dataSize, tmpPonitera, tmpPonitera, 1);
			VecDouble.scalarDiv(dataSize, resultPoniter, 1.0, tmpPonitera);
		}

		else {
			throw new Exception("un supported operations: " + activationType);
		}

		cuMemcpyDtoH(Pointer.to(y), resultPoniter, dataSize * Sizeof.DOUBLE);
		double[][] resultMat = GpuUtil.arrayToDoubleMatrix(y, m, n);
		y = null;
		tmpa = null;
		;

		GpuUtil.freeMemory(tmpPonitera, resultPoniter);
		VecDouble.shutdown();
		GpuUtil.shutDownGpuForVectorOpt();
		return resultMat;

	}
	

	


	/**
	 * 
	 * @param itemVector
	 * @param uniqueLabelSet
	 * @return
	 */
	public static Map<Double, Set<Double>> getItemIndexMapForMultiClassFileData(
			double[][] itemVector, Set<Double> uniqueLabelSet) {
		Map<Double, Set<Double>> resultMap = new HashMap<>();
		Set<Double> setForSingleItemIndex = null;
		for (Double item : uniqueLabelSet) {
			setForSingleItemIndex = new HashSet<>();
			for (int i = 0; i < itemVector.length; i++) {
				if (Double.compare(itemVector[i][0], item) == 0) {
					setForSingleItemIndex.add(item);
				}
			}
			resultMap.put(item, setForSingleItemIndex);
		}
		return resultMap;
	}

	
	/**
	 * 
	 * @param itemVector
	 * @param item
	 * @return
	 */
	public static Integer[] getItemIndexVectForMultiClassFileData(
			double[][] itemVector, double item) {

		Set<Integer> setForSingleItemIndex = null;

		setForSingleItemIndex = new HashSet<>();
		for (int i = 0; i < itemVector.length; i++) {
			if (Double.compare(itemVector[i][0], item) == 0) {
				setForSingleItemIndex.add(i);
			}
		}

		return (setForSingleItemIndex.toArray(new Integer[setForSingleItemIndex
				.size()]));
	}

	/**
	 * This is python numpy version of npGpu.add where the b is rescalled to
	 * fector or matrix on the basis of A
	 * 
	 * @todo
	 * @param a
	 * @param b
	 * @param isrescale_b
	 * @return
	 * @throws Exception
	 */

	public static double[][] add(double[][] a, double[][] b, boolean isrescale_b)
			throws Exception {
		// npGpu.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;
		double[][] c = new double[m][n];

		if ((b[0].length == 1) && b.length == 1) {

			return simplePointWiseOperationsWithScalarAsSecondOperand(a,
					b[0][0], SimpleMathOperations.ADD);
		} else if (b[0].length == 1 && b.length == m) {
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {

					c[i][j] = a[i][j] + b[i][0];

				}
			}
		} else if (b[0].length == n && b.length == 1) {
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {

					// Need to do unit testing
					c[i][j] = a[i][j] + b[0][j];

				}
			}
		} else {
			return simplePointWiseOperations(a, b, SimpleMathOperations.ADD);
		}

		return c;
	}

	
	/**
	 * 
	 * @param a
	 * @param b
	 * @param isrescale_b
	 * @return
	 * @throws Exception
	 */
	public static double[][] subtract(double[][] a, double[][] b,
			boolean isrescale_b) throws Exception {
		// npGpu.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;
		double[][] c = new double[m][n];

		if ((b[0].length == 1) && b.length == 1) {

			return simplePointWiseOperationsWithScalarAsSecondOperand(a,
					b[0][0], SimpleMathOperations.SUB);
		} else if (b[0].length == 1 && b.length == m) {
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {

					c[i][j] = a[i][j] - b[i][0];

				}
			}
		} else if (b[0].length == n && b.length == 1) {
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {

					c[i][j] = a[i][j] - b[0][j];

				}
			}
		}

		else {
			return simplePointWiseOperations(a, b, SimpleMathOperations.SUB);
		}

		return c;
	}
/**
 * 
 * @param a
 * @param b
 * @param isrescale_b
 * @return
 */
	public static double[][] subtract_old(double[][] a, double[][] b,
			boolean isrescale_b) {
		// np.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;
		double[][] c = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (isrescale_b) {
					if (b[0].length == 1 && b.length == m) {
						c[i][j] = a[i][j] - b[i][0];
					} else if (b.length == 1 && b[0].length == n) {
						c[i][j] = a[i][j] - b[0][j];
					}

					else if ((b[0].length == 1) && b.length == 1) {
						c[i][j] = a[i][j] - b[0][0];
					}

					else {
						c[i][j] = a[i][j] - b[i][j];
					}

				} else {
					c[i][j] = a[i][j] - b[i][j];
				}

			}
		}
		return c;
	}

	
	/**
	 * 
	 * @param a
	 * @param b
	 * @return
	 * @throws Exception
	 */
	public static double[][] add(double[][] a, double b) throws Exception {
		return simplePointWiseOperationsWithScalarAsSecondOperand(a, b,
				SimpleMathOperations.ADD);
	}

	/**
	 * @param a
	 *            matrix
	 * @param b
	 *            matrix
	 * @return c = a - b
	 */
	public static double[][] subtract(double[][] a, double[][] b)
			throws Exception {

		return simplePointWiseOperations(a, b, SimpleMathOperations.SUB);
	}

	public static void displayMatrix(final double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
	}

	/**
	 * Element wise subtraction
	 *
	 * @param a
	 *            scaler
	 * @param b
	 *            matrix
	 * @return c = a - b
	 * @throws Exception
	 */
	public static double[][] subtract(double a, double[][] b) throws Exception {
		return simplePointWiseOperationsWithScalarAsFirstOperand(a, b,
				SimpleMathOperations.SUB);
	}

	public static double[][] generateRandom(int rows, int columns,
			ProbablityDistributionTypes probType) throws IllegalAccessException {
		JCuda.setExceptionsEnabled(true);
		JCurand.setExceptionsEnabled(true);

		int n = rows * columns;
		curandGenerator generator = new curandGenerator();

		// Allocate n floats on host

		// Allocate n floats on host
		float hostData[] = new float[n];

		// Allocate n floats on device
		Pointer deviceData = new Pointer();
		cudaMalloc(deviceData, n * Sizeof.FLOAT);

		
		curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

		// Set seed
		curandSetPseudoRandomGeneratorSeed(generator, 1234);

		
		curandGenerateUniform(generator, deviceData, n);

		
		cudaMemcpy(Pointer.to(hostData), deviceData, n * Sizeof.FLOAT,
				cudaMemcpyDeviceToHost);

	

		// Cleanup
		curandDestroyGenerator(generator);
		cudaFree(deviceData);
		GpuUtil.shutDownGpuForVectorOpt();
		double[][] resultMatrix = GpuUtil.floatarrayToDoubleMatrix(hostData,
				rows, columns);
		hostData = null;
		return resultMatrix;

	}
/**
 * 
 * @param b
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] subtract(double[][] b, double a) throws Exception {

		return simplePointWiseOperationsWithScalarAsSecondOperand(b, a,
				SimpleMathOperations.SUB);
	}

/**
 * 	
 * @param B
 * @param C
 * @return
 */
	
	public static double[][] dott(double[][] B, double[][] C) {
		int m1 = B.length;
		int n1 = B[0].length;
		int m2 = C.length;
		int n2 = C[0].length;
		double[][] resultMatrix = new double[m1][n2];
		double[] matSinglElementTmp = null;
		CUdeviceptr tmpPonitera = null;
		CUdeviceptr tmpPoniterb = null;
		CUdeviceptr resultPoniter = null;
		CUdeviceptr pointerInGpu1 = null;
		CUdeviceptr pointerInGpu2 = null;
		try {
			matSinglElementTmp = new double[n1];
			GpuUtil.initGPUForVectorOpt();
			VecDouble.init();

			resultPoniter = new CUdeviceptr();
			// CUdeviceptr tmpPoniterb=null;
			pointerInGpu1 = new CUdeviceptr();
			cuMemAlloc(pointerInGpu1, n1 * Sizeof.DOUBLE);
			pointerInGpu2 = new CUdeviceptr();
			cuMemAlloc(pointerInGpu2, n1 * Sizeof.DOUBLE);

			cuMemAlloc(resultPoniter, n1 * Sizeof.DOUBLE);
			double[][] Ctrans = MatrixUtilGPU.T(C);
			// System.out.println(Arrays.deepToString(Ctrans));
			for (int i = 0; i < m1; i++) {
				tmpPonitera = GpuUtil.getGpuMemoryPointer(B[i], n1,
						pointerInGpu1);
				for (int j = 0; j < n2; j++) {

					tmpPoniterb = GpuUtil.getGpuMemoryPointer(Ctrans[j], n1,
							pointerInGpu2);
					VecDouble.mul(n1, resultPoniter, tmpPonitera, tmpPoniterb);
					cuMemcpyDtoH(Pointer.to(matSinglElementTmp), resultPoniter,
							n1 * Sizeof.DOUBLE);
					resultMatrix[i][j] = MatrixUtil.sum(matSinglElementTmp);
				}
			}
			return resultMatrix;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			resultMatrix = null;
			matSinglElementTmp = null;
			GpuUtil.freeMemory(resultPoniter, pointerInGpu1, pointerInGpu2);
			VecDouble.shutdown();
			GpuUtil.shutDownGpuForVectorOpt();
		}
		return resultMatrix;
	}

	/**
	 * This metod uses cublas library for matrix dot product.
	 * @param handle
	 * @param rows
	 * @param columns
	 * @param k
	 * @param A
	 * @param lda
	 * @param B
	 * @param ldb
	 * @return
	 */
	private static float[] dot_cublas(cublasHandle handle, int rows,
			int columns, int k, float A[], int lda, float B[], int ldb) {
		Pointer dA = new Pointer();
		Pointer dB = new Pointer();
		Pointer dC = new Pointer();

		cudaMalloc(dA, lda * k * Sizeof.FLOAT);
		cudaMalloc(dB, (ldb * columns) * Sizeof.FLOAT);
		cudaMalloc(dC, rows * columns * Sizeof.FLOAT);
		cublasSetVector((lda * k), Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
		cublasSetVector((ldb * columns), Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);
		float[] resultMatrixVect = new float[rows * columns];
		Pointer zero = Pointer.to(new float[] { 0.0f });
		Pointer one = Pointer.to(new float[] { 1.0f });

		JCublas2.cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, columns,
				k, one, dA, lda, dB, ldb, zero, dC, rows);
		cublasGetVector(rows * columns, Sizeof.FLOAT, dC, 1,
				Pointer.to(resultMatrixVect), 1);

		cudaFree(dA);
		cudaFree(dB);
		cudaFree(dC);
		cudaFree(zero);
		cudaFree(one);

		return resultMatrixVect;
	}

/**
 * 	
 * @param x
 * @return
 */
	
	public static double max(double[][] x) {
		double max = Double.MIN_VALUE;
		JCublas.cublasInit();
		int dataSize = x.length * x[0].length;
		Pointer dA = new Pointer();
		cudaMalloc(dA, x.length * x[0].length * Sizeof.DOUBLE);
		double[] xTmp = GpuUtil.matrixtoArray(x, x.length, x[0].length);
		cublasSetVector(dataSize, Sizeof.DOUBLE, Pointer.to(xTmp), 1, dA, 1);

		int imax = JCublas.cublasIcamax(dataSize, dA, 1);

		max = xTmp[imax - 1];

		cudaFree(dA);

		return max;
	}
/**
 * 
 * @param x
 * @return
 */
	public static double min(double[][] x) {
		double max = Double.MIN_VALUE;
		JCublas.cublasInit();
		int dataSize = x.length * x[0].length;
		Pointer dA = new Pointer();
		cudaMalloc(dA, x.length * x[0].length * Sizeof.DOUBLE);
		double[] xTmp = GpuUtil.matrixtoArray(x, x.length, x[0].length);
		cublasSetVector(dataSize, Sizeof.DOUBLE, Pointer.to(xTmp), 1, dA, 1);

		int imax = JCublas.cublasIcamin(dataSize, dA, 1);

		max = xTmp[imax - 1];

		cudaFree(dA);

		return max;
	}
/**
 * Matrix dot product using culblas library
 * @param handle
 * @param rows
 * @param columns
 * @param k
 * @param A
 * @param lda
 * @param B
 * @param ldb
 * @return
 */
	private static double[] dot_cublas(cublasHandle handle, int rows,
			int columns, int k, double A[], int lda, double B[], int ldb) {
		Pointer dA = new Pointer();
		Pointer dB = new Pointer();
		Pointer dC = new Pointer();

		cudaMalloc(dA, lda * k * Sizeof.DOUBLE);
		cudaMalloc(dB, (ldb * columns) * Sizeof.DOUBLE);
		cudaMalloc(dC, rows * columns * Sizeof.DOUBLE);
		cublasSetVector((lda * k), Sizeof.DOUBLE, Pointer.to(A), 1, dA, 1);
		cublasSetVector((ldb * columns), Sizeof.DOUBLE, Pointer.to(B), 1, dB, 1);
		double[] resultMatrixVect = new double[rows * columns];
		Pointer zero = Pointer.to(new double[] { 0.0 });
		Pointer one = Pointer.to(new double[] { 1.0 });
		// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, columns, k, one,
		// dA,
		// rows, dB, rows, zero, dC, (rows));
		JCublas2.cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, columns,
				k, one, dA, lda, dB, ldb, zero, dC, rows);
		cublasGetVector(rows * columns, Sizeof.DOUBLE, dC, 1,
				Pointer.to(resultMatrixVect), 1);

		cudaFree(dA);
		cudaFree(dB);
		cudaFree(dC);
		cudaFree(zero);
		cudaFree(one);

		return resultMatrixVect;
	}
/**
 * This method is for dot products between two LargeDoubleMatrix which is actually reside on hard dist instead of main memory
 * @param B
 * @param C
 * @param resultMatrix
 * @throws InterruptedException
 */
	public static void dot(LargeDoubleMatrix B, LargeDoubleMatrix C,
			LargeDoubleMatrix resultMatrix) throws InterruptedException {
		int m1 = B.height();
		int n1 = B.width();
		int m2 = C.height();
		int n2 = C.width();

		cublasHandle handle = new cublasHandle();
		cublasCreate(handle);

		try {

			float[] Bvect = GpuUtil.LargeDoubleMatrixtoFloatArray(B, m1, n1);
			float[] Cvect = GpuUtil.LargeDoubleMatrixtoFloatArray(C, m2, n2);

			float[] resultMatrixVect = dot_cublas(handle, m1, n2, n1, Bvect,
					m1, Cvect, m2);

			Bvect = null;
			Cvect = null;
			GpuUtil.floatArrayToLargeDoubleMatrix(resultMatrixVect, m1, n2,
					resultMatrix);
			resultMatrixVect = null;
			cublasDestroy(handle);
			GpuUtil.shutDownGpuForVectorOpt();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {

		}
		// return resultMatrix;
	}

	public static double[][] dot(double[][] B, double[][] C) throws Exception {
		int m1 = B.length;
		int n1 = B[0].length;
		int m2 = C.length;
		int n2 = C[0].length;

		cublasHandle handle = new cublasHandle();
		cublasCreate(handle);

		try {

			double[] Bvect = GpuUtil.matrixtoArray(B, m1, n1);
			double[] Cvect = GpuUtil.matrixtoArray(C, m2, n2);

			double[] resultMatrixVect = dot_cublas(handle, m1, n2, n1, Bvect,
					m1, Cvect, m2);

			Bvect = null;
			Cvect = null;
			double[][] resultMatrix = GpuUtil.arrayToDoubleMatrix(
					resultMatrixVect, m1, n2);
			resultMatrixVect = null;
			cublasDestroy(handle);
			GpuUtil.shutDownGpuForVectorOpt();
			return resultMatrix;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			throw e;
		} finally {

		}
		// return resultMatrix;
	}

	public static void dott(LargeDoubleMatrix B, LargeDoubleMatrix C,
			LargeDoubleMatrix resultMatrix) {
		int m1 = B.height();
		int n1 = B.width();
		int m2 = C.height();
		int n2 = C.width();
		// double[][] resultMatrix = new double[m1][n2];
		double[] matSinglElementTmp = null;
		CUdeviceptr tmpPonitera = null;
		CUdeviceptr tmpPoniterb = null;
		CUdeviceptr resultPoniter = null;
		CUdeviceptr pointerInGpu1 = null;
		CUdeviceptr pointerInGpu2 = null;
		try {
			matSinglElementTmp = new double[n1];
			GpuUtil.initGPUForVectorOpt();
			VecDouble.init();

			resultPoniter = new CUdeviceptr();

			cuMemAlloc(resultPoniter, n1 * Sizeof.DOUBLE);
			pointerInGpu1 = new CUdeviceptr();
			cuMemAlloc(pointerInGpu1, n1 * Sizeof.DOUBLE);
			pointerInGpu2 = new CUdeviceptr();
			cuMemAlloc(pointerInGpu2, n1 * Sizeof.DOUBLE);
			// System.out.println(Arrays.deepToString(Ctrans));
			for (int i = 0; i < m1; i++) {
				tmpPonitera = GpuUtil.getGpuMemoryPointer(B.getRow(i), n1,
						pointerInGpu1);
				for (int j = 0; j < n2; j++) {

					tmpPoniterb = GpuUtil.getGpuMemoryPointer(C.getColumn(j),
							n1, pointerInGpu2);
					VecDouble.mul(n1, resultPoniter, tmpPonitera, tmpPoniterb);
					cuMemcpyDtoH(Pointer.to(matSinglElementTmp), resultPoniter,
							n1 * Sizeof.DOUBLE);
					resultMatrix.set(i, j,
							getSumOfAllElements(matSinglElementTmp));
				}
			}
			// return resultMatrix;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			resultMatrix = null;
			matSinglElementTmp = null;
			GpuUtil.freeMemory(resultPoniter, pointerInGpu1, pointerInGpu2);
			VecDouble.shutdown();
			GpuUtil.shutDownGpuForVectorOpt();
		}
		// return resultMatrix;
	}

	public static void dotRange(LargeDoubleMatrix B, LargeDoubleMatrix C,
			LargeDoubleMatrix resultMatrix, int intialIndex, int finalIndex,
			CUdeviceptr resultPoniter, CUdeviceptr pointerInGpu1,
			CUdeviceptr pointerInGpu2) {
		int m1 = B.height();
		int n1 = B.width();
		int m2 = C.height();
		int n2 = C.width();
		// double[][] resultMatrix = new double[m1][n2];
		double[] matSinglElementTmp = null;
		CUdeviceptr tmpPonitera = null;
		CUdeviceptr tmpPoniterb = null;
		// CUdeviceptr resultPoniter = null;
		// final CUdeviceptr pointerInGpu1 = null;
		// final CUdeviceptr pointerInGpu2 = null;
		try {
			matSinglElementTmp = new double[n1];

			// System.out.println(Arrays.deepToString(Ctrans));
			for (int i = intialIndex; i < finalIndex; i++) {
				tmpPonitera = GpuUtil.getGpuMemoryPointer(B.getRow(i), n1,
						pointerInGpu1);
				for (int j = 0; j < n2; j++) {

					tmpPoniterb = GpuUtil.getGpuMemoryPointer(C.getColumn(j),
							n1, pointerInGpu2);
					VecDouble.mul(n1, resultPoniter, tmpPonitera, tmpPoniterb);
					cuMemcpyDtoH(Pointer.to(matSinglElementTmp), resultPoniter,
							n1 * Sizeof.DOUBLE);
					resultMatrix.set(i, j,
							getSumOfAllElements(matSinglElementTmp));
				}
			}
			// return resultMatrix;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			resultMatrix = null;
			matSinglElementTmp = null;
			// GpuUtil.freeMemory(resultPoniter, pointerInGpu1, pointerInGpu2);

		}
		// return resultMatrix;
	}
	public static double getSumOfAllElements(double[] vector)
			throws InterruptedException {
		double sumOfAllElements = 0.0;

		final int length = vector.length;
		final int threads = length > IPredictionModelConstants.ARRAY_LENTH_PER_THREAD ? length
				/ IPredictionModelConstants.ARRAY_LENTH_PER_THREAD
				: 1;

		return (DoubleArraySumMultiThreaded.parallelSum(vector, threads));
	}

	public static double sumRange(double[] a, int min, int max) {
		double result = 0;
		for (int i = min; i < max; i++) {
			result += a[i];
		}
		return result;
	}


	public static float[][] dot(float[][] B, float[][] C) throws Exception {
		int m1 = B.length;
		int n1 = B[0].length;
		int m2 = C.length;
		int n2 = C[0].length;

		cublasHandle handle = new cublasHandle();
		cublasCreate(handle);

		try {

			float[] Bvect = GpuUtil.matrixtoArray(B, m1, n1);
			float[] Cvect = GpuUtil.matrixtoArray(C, m2, n2);

			float[] resultMatrixVect = dot_cublas(handle, m1, n2, n1, Bvect,
					m1, Cvect, m2);

			Bvect = null;
			Cvect = null;
			float[][] resultMatrix = GpuUtil.arrayToFloatMatrix(
					resultMatrixVect, m1, n2);
			resultMatrixVect = null;
			cublasDestroy(handle);
			GpuUtil.shutDownGpuForVectorOpt();

			return resultMatrix;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			throw e;
		} finally {

		}
		// return resultMatrix;
	}

	/**
	 * get some along some axis
	 * 
	 * @param a
	 * @param axis
	 * @return
	 */
	public static double[][] sum(double[][] a, int axis) {
		int m1 = a.length;
		int n1 = a[0].length;
		double[][] c = null;
		if (axis == 1) {
			c = new double[m1][1];
			for (int i = 0; i < m1; i++) {
				double sum = 0.0;
				for (int j = 0; j < n1; j++) {

					sum = sum + a[i][j];

				}
				c[i][0] = sum;
			}
		} else if (axis == 0) {
			c = new double[1][n1];
			for (int j = 0; j < n1; j++) {
				double sum = 0.0;
				for (int i = 0; i < m1; i++) {

					sum = sum + a[i][j];

				}
				c[0][j] = sum;
			}
		}

		return c;
	}

	/**
	 * Element wise multiplication
	 *
	 * @param a
	 *            matrix
	 * @param x
	 *            matrix
	 * @return y = a * x
	 */
	public static double[] toVector(double[][] x) {
		int m = x.length;
		double[] a = new double[m];
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < 1; i++) {
				a[j] = x[j][0];
			}
		}
		return a;
	}

	public static double[][] createBinaryVector(double[][] x, double label) {
		int m = x.length;
		int n = x[0].length;
		double[][] a = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (Double.compare(x[i][j], label) == 0) {
					a[i][j] = 1;
				} else {
					a[i][j] = 0;
				}
			}
		}
		return a;
	}

	public static double[][] multiply(double[][] x, double[][] a)
			throws Exception {

		return simplePointWiseOperations(x, a, SimpleMathOperations.MUL);

	}

	public static float[][] multiply(float[][] x, float[][] a) throws Exception {

		return simplePointWiseOperations(x, a, SimpleMathOperations.MUL);

	}

	public static double[][] multiply(double[][] x, double a) throws Exception {

		return simplePointWiseOperationsWithScalarAsSecondOperand(x, a,
				SimpleMathOperations.MUL);
	}

	/**
	 * Element wise multiplication
	 *
	 * @param a
	 *            matrix
	 * @param x
	 *            scaler
	 * @return y = a * x
	 */
	public static double[][] multiply(double x, double[][] a) {
		int m = a.length;
		int n = a[0].length;

		double[][] y = new double[m][n];
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				y[j][i] = a[j][i] * x;
			}
		}
		return y;
	}

	/**
	 * Element wise power
	 *
	 * @param x
	 *            matrix
	 * @param a
	 *            scaler
	 * @return y
	 */
	public static double[][] power(double[][] x, int a) {
		int m = x.length;
		int n = x[0].length;

		double[][] y = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				y[i][j] = Math.pow(x[i][j], a);
			}
		}
		return y;
	}

	/**
	 * 
	 * @param x
	 * @return
	 */
	public static double[][] exp(double[][] x) {
		int m = x.length;
		int n = x[0].length;

		double[][] y = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				y[i][j] = Math.exp(x[i][j]);
			}
		}
		return y;
	}

	/**
	 * @param a
	 *            matrix
	 * @return shape of matrix a
	 */
	public static String shape(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		String Vshape = "(" + m + "," + n + ")";
		return Vshape;
	}

	public static double[] shapeDim(double[][] a) {
		int m = a.length;
		int n = a[0].length;

		return new double[] { m, n };
	}

	public static int[][] reshape(int[][] nums, int r, int c) {
		int totalElements = nums.length * nums[0].length;
		if (totalElements != r * c || totalElements % r != 0) {
			return nums;
		}
		final int[][] result = new int[r][c];
		int newR = 0;
		int newC = 0;
		for (int i = 0; i < nums.length; i++) {
			for (int j = 0; j < nums[i].length; j++) {
				result[newR][newC] = nums[i][j];
				newC++;
				if (newC == c) {
					newC = 0;
					newR++;
				}
			}
		}
		return result;
	}

	public static double[][] reshape(double[][] nums, int r, int c) {
		int totalElements = nums.length * nums[0].length;
		if (totalElements != r * c || totalElements % r != 0) {
			return nums;
		}
		final double[][] result = new double[r][c];
		int newR = 0;
		int newC = 0;
		for (int i = 0; i < nums.length; i++) {
			for (int j = 0; j < nums[i].length; j++) {
				result[newR][newC] = nums[i][j];
				newC++;
				if (newC == c) {
					newC = 0;
					newR++;
				}
			}
		}
		return result;
	}

	public static double[][] cloneArray(double[][] src) {
		int length = src.length;
		double[][] target = new double[length][src[0].length];
		for (int i = 0; i < length; i++) {
			System.arraycopy(src[i], 0, target[i], 0, src[i].length);
		}
		return target;
	}

	public static double[][] replaceNegativeValuesOnOtherArrayBasis(
			double[][] a, double[][] other, double valueToBeReplaced) {
		int m = a.length;
		int n = a[0].length;

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				if (other[i][j] < 0) {
					a[i][j] = valueToBeReplaced;
				} else {
					a[i][j] = a[i][j];
				}
		return a;
	}

	public static double[][] createBooleanMatrixForSomeThreshold(double[][] a,
			double threshold) {
		int m = a.length;
		int n = a[0].length;
		double[][] newMatrix = new double[m][n];
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				if (a[i][j] < threshold) {
					newMatrix[i][j] = 1;
				} else {
					newMatrix[i][j] = 0.0;
				}
		return newMatrix;
	}

	public static double[][] createMatrixWithConstantValue(int m, int n,
			double constantValue) {

		double[][] a = new double[m][n];
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++) {
				a[i][j] = constantValue;
			}
		return a;
	}

	public static double[][] createIdentity(int m, int n) {

		double[][] a = new double[m][n];
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++) {
				if (i == j) {
					a[i][j] = 1;
				} else {
					a[i][j] = 0.0;
				}

			}
		return a;
	}

	/**
	 * @param a
	 *            matrix
	 * @return sigmoid of matrix a
	 */
	public static double[][] sigmoid(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] z = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				z[i][j] = (1.0 / (1 + Math.exp(-a[i][j])));
			}
		}
		return z;
	}

	public static double[][] swish(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] z = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				z[i][j] = (a[i][j] / (1 + Math.exp(-a[i][j])));
			}
		}
		return z;
	}

	public static double[][] tanh(double[][] a) throws Exception {
		/*
		 * int m = a.length; int n = a[0].length; double[][] z = new
		 * double[m][n]; double tmpExpPositve=0.0,tmpExpNegative=0.0; for (int i
		 * = 0; i < m; i++) { for (int j = 0; j < n; j++) {
		 * tmpExpPositve=Math.exp(a[i][j]); tmpExpNegative=Math.exp(-a[i][j]);
		 * z[i][j] = ((tmpExpPositve - tmpExpNegative) / (tmpExpPositve +
		 * tmpExpNegative)); } }
		 */
		return MatrixUtilGPU.activations(a, DeepNNActivations.TANH);

	}

	public static double[][] ktest(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] z = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				// z[i][j] =m*((Math.exp(a[i][j]) - Math.exp(-a[i][j])) /
				// (Math.exp(a[i][j]) +
				// Math.exp(-a[i][j])));
				// z[i][j] = ((Math.exp(a[i][j]) - Math.exp(-a[i][j])) /
				// (Math.exp(a[i][j]) +
				// Math.exp(-a[i][j]))/a[i][j]);
				z[i][j] = ((Math.exp(a[i][j])) / (Math.exp(a[i][j]) + Math
						.exp(-a[i][j])));
			}
		}
		return z;
	}

	public static double[][] relu(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] z = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				z[i][j] = Math.max(0.0, a[i][j]);
			}
		}
		return z;
	}

	public static float[][] divide(float[][] x, float a) throws Exception {

		return (simplePointWiseOperationsWithScalarAsSecondOperand(x, a,
				SimpleMathOperations.DIV));
	}

	public static double[][] divide(double[][] x, double a) throws Exception {

		return simplePointWiseOperationsWithScalarAsSecondOperand(x, a,
				SimpleMathOperations.DIV);
	}

	public static double[][] divide(double a, double[][] x) throws Exception {
		return simplePointWiseOperationsWithScalarAsFirstOperand(a, x,
				SimpleMathOperations.DIV);

	}

	public static double[][] divide(double[][] a, double[][] b)
			throws Exception {

		return simplePointWiseOperations(a, b, SimpleMathOperations.DIV);

	}

	/**
	 * Normalized data according to row and column axis=0 means rowwise, axis=1
	 * means column wise
	 * 
	 * @param a
	 * @param axis
	 * @return
	 */
	public static double[][] normalized(double[][] a, int axis) {
		double[][] meanNormalized = null;
		meanNormalized = new double[a.length][a[0].length];
		double[] mean = null;
		double[] sigma = null;
		if (axis == 1) {
			mean = new double[a.length];
			sigma = new double[a.length];
			for (int i = 0; i < a.length; i++) {
				double rowSum = 0;
				for (int j = 0; j < a[0].length; j++) {
					rowSum = rowSum + a[i][j];
					sigma[i] = sigma[i] + (a[i][j] * a[i][j]);
				}

				mean[i] = rowSum / a[0].length;

				for (int j = 0; j < a[0].length; ++j) {

					sigma[i] = sigma[i]
							+ ((a[i][j] - mean[i]) * (a[i][j] - mean[i]));
				}

				sigma[i] = Math.sqrt(sigma[i] / a[0].length);
				if (rowSum == 0) {
					continue;
				}
				for (int j = 0; j < a[0].length; j++) {
					if (sigma[i] != 0) {
						meanNormalized[i][j] = (a[i][j] - mean[i]) / sigma[i];
					} else {
						meanNormalized[i][j] = (a[i][j] - mean[i]);
					}

				}

			}

		} else if (axis == 0) {
			mean = new double[a[0].length];
			sigma = new double[a[0].length];
			for (int i = 0; i < a[0].length; ++i) {
				double colSum = 0;
				for (int j = 0; j < a.length; ++j) {
					colSum = colSum + a[j][i];

				}
				mean[i] = colSum / a.length;
				// to claculate sigma
				for (int j = 0; j < a.length; ++j) {

					sigma[i] = sigma[i]
							+ ((a[j][i] - mean[i]) * (a[j][i] - mean[i]));
				}

				sigma[i] = Math.sqrt(sigma[i] / a.length);
				// Skip columns whose sume is zero.
				if (colSum == 0)
					continue;
				for (int j = 0; j < a.length; ++j) {
					if (sigma[i] != 0) {
						meanNormalized[j][i] = (a[j][i] - mean[i]) / sigma[i];
					} else {
						meanNormalized[j][i] = (a[j][i] - mean[i]);
					}

				}

			}
		}
		return meanNormalized;

	}

	/**
	 * Element wise division
	 *
	 * @param A
	 *            matrix
	 * @param Y
	 *            matrix
	 * @param batch_size
	 *            scaler
	 * @return loss
	 * @throws Exception
	 */

	public static double[][] softmax(double[][] z, int axis) throws Exception {
		double[][] zout = null;
		double[] sumVector = null;
		int numberOfRows = z.length;
		int numberOfColumns = z[0].length;
		if (axis == 0) {
			z = MatrixUtil.subtract(z, MatrixUtil.max(z));
			zout = new double[numberOfRows][numberOfColumns];
			sumVector = new double[numberOfColumns];
			for (int j = 0; j < numberOfColumns; j++) {
				double sum = 0.0;
				for (int i = 0; i < numberOfRows; i++) {
					sum += Math.exp(z[i][j]);
				}
				sumVector[j] = sum;
			}
			for (int j = 0; j < numberOfColumns; j++) {
				for (int i = 0; i < numberOfRows; i++) {
					zout[i][j] = Math.exp(z[i][j]) / sumVector[j];
				}
			}
		} else if (axis == 1) {
			// @todo later
			// zout=softmax_old(z);
			throw new Exception("method not implemented");
		}

		return zout;
	}

	public static double[][] softmax_old(double[][] z) {
		double[][] zout = new double[z.length][z[0].length];
		double sum = 0.;
		for (int i = 0; i < z.length; i++) {
			for (int j = 0; j < z[0].length; j++) {
				sum += Math.exp(z[i][j]);
			}
		}
		for (int i = 0; i < z.length; i++) {
			for (int j = 0; j < z[0].length; j++) {
				zout[i][j] = Math.exp(z[i][j]) / sum;
			}
		}
		return zout;
	}

	public static void print(String val) {
		System.out.println(val);
	}

	public static void printshapes(double[][]... list) {
		for (double[][] obj : list)
			System.out.println(obj + ": " + MatrixUtil.shape(obj));

	}

	public static int getIndexOfMax(double[] a) {

		int dataSize = a.length;
		Pointer dA = new Pointer();
		cudaMalloc(dA, a.length * Sizeof.DOUBLE);

		cublasSetVector(dataSize, Sizeof.DOUBLE, Pointer.to(a), 1, dA, 1);

		int imax = JCublas.cublasIcamax(dataSize, dA, 1);

		cudaFree(dA);
		return (imax - 1);
	}

	public static int getIndexOfMin(double[] a) {

		int dataSize = a.length;
		Pointer dA = new Pointer();
		cudaMalloc(dA, a.length * Sizeof.DOUBLE);

		cublasSetVector(dataSize, Sizeof.DOUBLE, Pointer.to(a), 1, dA, 1);

		int imax = JCublas.cublasIcamin(dataSize, dA, 1);

		cudaFree(dA);
		return (imax - 1);
	}

	public static void assertion(double[][] a, double[][] b) {
		// assert (dZ.shape == Z.shape);
		// np.printshapes(a,b);
		try {
			if (!(a.length == b.length) && (a[0].length == b[0].length)) {
				MatrixUtil.print("a: " + MatrixUtil.shape(a) + "b: "
						+ MatrixUtil.shape(b));
			}

			assert ((a.length == b.length) && (a[0].length == b[0].length));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			MatrixUtil.print("a: " + MatrixUtil.shape(a) + "b: "
					+ MatrixUtil.shape(b));
			throw e;
		}

	}

	public static Map<DeepNNKeysForMaps, Object> createOneHotVectorMatrix(
			double[][] Y) {
		Map<DeepNNKeysForMaps, Object> resultMap = new HashMap<>();
		Set<Double> labels = new HashSet<>();

		double[][] oneHotIdentityMarix = null;
		for (int i = 0; i < Y.length; i++) {
			if (!labels.contains(Y[i][0])) {
				labels.add(Y[i][0]);// / making set of unique lables
			}

		}

		oneHotIdentityMarix = MatrixUtil.createIdentity(labels.size(),
				labels.size());
		double[][] Y_OneHot = new double[Y.length][labels.size()];
		Map<Double, double[]> labeltoOnhotVectorMap = new HashMap<>();
		List<Double> list = new ArrayList(labels);
		Collections.sort(list);
		Iterator<Double> value = list.iterator();
		int count = 0;
		while (value.hasNext()) {
			double tmpValue = value.next();
			labeltoOnhotVectorMap.put(tmpValue, oneHotIdentityMarix[count]);
			// System.out.println("label: "+tmpValue+" corresponding on
			// hot:"+Arrays.toString(oneHotIdentityMarix[count]));
			count++;
		}

		for (int i = 0; i < Y.length; i++) {
			Y_OneHot[i] = labeltoOnhotVectorMap.get(Y[i][0]);

		}
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_LABEL_TO_ONE_HOT,
				labeltoOnhotVectorMap);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_ONE_HOT_MATRIX, Y_OneHot);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_NUMBER_OF_CLASSES,
				labeltoOnhotVectorMap.keySet().size());
		return resultMap;
	}

	/**
	 * 
	 * @param labeltoOnhotVectorMap
	 * @param predictedOneHotVector
	 * @return
	 */
	public static double getLabelFromOneHotMatrix(
			Map<Double, double[]> labeltoOnhotVectorMap, int predictedIndx) {
		double label = -1;
		double[] tmp = null;
		for (Double labelInMatr : labeltoOnhotVectorMap.keySet()) {
			tmp = labeltoOnhotVectorMap.get(labelInMatr);
			if (Double.compare(tmp[predictedIndx], 1.0) == 0) {

				return labelInMatr;
			}

		}

		return label;
	}

}