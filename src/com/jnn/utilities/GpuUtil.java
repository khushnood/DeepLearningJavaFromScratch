package com.jnn.utilities;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemAllocHost;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemFreeHost;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
/**
 *  Copyright 2020
 *   
 * @author Khushnood Abbas
 * @email khushnood.abbas@gmail.com
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
public class GpuUtil {

	static {
		if (MatrixUtil.isGpuEnabled) {
			// init();
		}
	}

	private final CUcontext context;
	private final CUmodule module;
	private final CUfunction function;
	private final CUstream stream;

	private int localDataSize;
	private CUdeviceptr deviceData;
	private CUdeviceptr deviceSum;
	private Pointer hostSumFromDevice;

	GpuUtil(CUcontext context, String ptxFileName) {
		this.context = context;

		cuCtxSetCurrent(context);

		module = new CUmodule();
		cuModuleLoad(module, ptxFileName);

		function = new CUfunction();
		cuModuleGetFunction(function, module, "reduceKernel");

		stream = new CUstream();
		cuStreamCreate(stream, 0);
	}
/**
 * 
 * @param localDataSize
 * @param gridSize
 * @param blockSize
 */
	void initMemory(int localDataSize, int gridSize, int blockSize) {
		cuCtxSetCurrent(context);

		this.localDataSize = localDataSize;

		deviceData = new CUdeviceptr();
		cuMemAlloc(deviceData, localDataSize * Sizeof.FLOAT);

		deviceSum = new CUdeviceptr();
		cuMemAlloc(deviceSum, gridSize * blockSize * Sizeof.FLOAT);

		hostSumFromDevice = new CUdeviceptr();
		cuMemAllocHost(hostSumFromDevice, gridSize * blockSize * Sizeof.FLOAT);
	}

	void freeMemory() {
		cuCtxSetCurrent(context);

		cuMemFree(deviceData);
		cuMemFree(deviceSum);
		cuMemFreeHost(hostSumFromDevice);
	}

	void shutdown() {
		cuCtxSetCurrent(context);
		cuStreamDestroy(stream);
		cuCtxDestroy(context);
	}

	static CUdevice device = null;
	static CUcontext contextForVec = null;
	
	public static void initGPUForVectorOpt() {
		JCuda.cudaDeviceReset();
		JCudaDriver.setExceptionsEnabled(true);
		cuInit(0);
		device = new CUdevice();
		cuDeviceGet(device, 0);
		contextForVec = new CUcontext();
		
		cuCtxCreate(contextForVec, 0, device);
		cuCtxSetCurrent(contextForVec);
		
       		
	}

	public static void shutDownGpuForVectorOpt() {
		/*
		 * cuCtxSetCurrent(contextForVec);
		 * 
		 * cuCtxDestroy(contextForVec);
		 */
		JCuda.cudaDeviceReset();
	}
	
	
	
	public static CUdeviceptr getGpuMemoryPointer(double[] data, int sizeOfData) {
		CUdeviceptr pointerInGpu = new CUdeviceptr();
		cuMemAlloc(pointerInGpu, sizeOfData * Sizeof.DOUBLE);
		cuMemcpyHtoD(pointerInGpu, Pointer.to(data), sizeOfData * Sizeof.DOUBLE);
		return pointerInGpu;
	}

	public static CUdeviceptr getGpuMemoryPointer(double[] data, int sizeOfData, CUdeviceptr pointerInGpu) {
		if (pointerInGpu == null) {
			pointerInGpu = new CUdeviceptr();
			cuMemAlloc(pointerInGpu, sizeOfData * Sizeof.DOUBLE);
		}

		cuMemcpyHtoD(pointerInGpu, Pointer.to(data), sizeOfData * Sizeof.DOUBLE);
		return pointerInGpu;
	}

	public static CUdeviceptr getGpuMemoryPointer(float[] data, int sizeOfData, CUdeviceptr pointerInGpu) {
		if (pointerInGpu == null) {
			pointerInGpu = new CUdeviceptr();
			cuMemAlloc(pointerInGpu, sizeOfData * Sizeof.FLOAT);
		}

		cuMemcpyHtoD(pointerInGpu, Pointer.to(data), sizeOfData * Sizeof.FLOAT);
		return pointerInGpu;
	}

	public static void freeMemory(CUdeviceptr... list) {
		for (CUdeviceptr ptr : list) {
			if (ptr != null) {
				cuMemFree(ptr);
			}

		}
	}

	public static double[] LargeDoubleMatrixtoArray(LargeDoubleMatrix matrix, int rows, int columns) {
		double[] resultVector = new double[rows * columns];
		int count = 0;
		for (int i = 0; i < matrix.height(); i++) {
			for (int j = 0; j < matrix.width(); j++) {

				resultVector[indexColMajor(i, j, rows)] = matrix.get(i, j);

			}

		}
		return resultVector;
	}
	public static double[] matrixtoArray(double[][] matrix, int rows, int columns) {
		double[] resultVector = new double[rows * columns];
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {

				resultVector[indexColMajor(i, j, rows)] = matrix[i][j];

			}

		}
		return resultVector;
	}
	
	public static float[] matrixtoArray(float[][] matrix, int rows, int columns) {
		float[] resultVector = new float[rows * columns];
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {

				resultVector[indexColMajor(i, j, rows)] = matrix[i][j];

			}

		}
		return resultVector;
	}
	public static float[] LargeDoubleMatrixtoFloatArray(LargeDoubleMatrix matrix, int rows, int columns) {
		float[] resultVector = new float[rows * columns];
		int count = 0;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {

				resultVector[indexColMajor(i, j, rows)] = (float) matrix.get(i, j);
				count++;
			}

		}
		return resultVector;
	}

	public static int indexColMajor(int rowIndex, int columnIndex, int rows) {
		return (columnIndex * rows) + (rowIndex);
	}

	public static int indexRowMajor(int rowIndex, int columnIndex, int columns) {
		return columnIndex + (columns * rowIndex);
	}

	public static void arrayToLargeDoubleMatrix(double[] vector, int rows, int columns, LargeDoubleMatrix resultMatrix)
			throws IllegalAccessException {

		if (vector.length != (resultMatrix.height() * resultMatrix.width())) {
			throw new IllegalAccessException("Dimension does not mathc");
		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {

				resultMatrix.set(i, j, vector[indexColMajor(i, j, rows)]);

			}

		}
	}
	public static double[][] arrayToDoubleMatrix(double[] vector, int rows, int columns)
			throws IllegalAccessException {

		if (vector.length != (rows*columns)) {
			throw new IllegalAccessException("Dimension does not mathc");
		}
		double[][] resultMatrix=new double[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {

				resultMatrix[i][j]= vector[indexColMajor(i, j, rows)];

			}

		}
		return resultMatrix;
	}
	public static float[][] arrayToFloatMatrix(float[] vector, int rows, int columns)
			throws IllegalAccessException {

		if (vector.length != (rows*columns)) {
			throw new IllegalAccessException("Dimension does not mathc");
		}
		float[][] resultMatrix=new float[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {

				resultMatrix[i][j]= vector[indexColMajor(i, j, rows)];

			}

		}
		return resultMatrix;
	}
	public static double[][] floatarrayToDoubleMatrix(float[] vector, int rows, int columns)
			throws IllegalAccessException {

		if (vector.length != (rows*columns)) {
			throw new IllegalAccessException("Dimension does not mathc");
		}
		double[][] resultMatrix=new double[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {

				resultMatrix[i][j]= (double)vector[indexColMajor(i, j, rows)];

			}

		}
		return resultMatrix;
	}
	public static void floatArrayToLargeDoubleMatrix(float[] vector, int rows, int columns,
			LargeDoubleMatrix resultMatrix) throws IllegalAccessException {

		if (vector.length != (resultMatrix.height() * resultMatrix.width())) {
			throw new IllegalAccessException("Dimension does not mathc");
		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {

				resultMatrix.set(i, j, (double) vector[indexColMajor(i, j, rows)]);

			}

		}
	}
}
