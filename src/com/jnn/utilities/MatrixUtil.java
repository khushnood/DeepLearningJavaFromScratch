package com.jnn.utilities;

import java.io.IOException;
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

import com.jnn.consts.IPredictionModelConstants;
import com.jnn.enums.DeepNNActivations;
import com.jnn.enums.DeepNNKeysForMaps;
import com.jnn.enums.ProbablityDistributionTypes;
import com.jnn.enums.StandardMathFunctions;
import com.jnn.utilities.Stopwatch;
import com.jnn.utilities.LargeDoubleMatrix;

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


public class MatrixUtil {

	private static Random random;
	private static long seed;
	public static boolean isGpuEnabled = false;

	public static boolean isGpuEnabled() {
		return isGpuEnabled;
	}

	public static void setGpuEnabled(boolean isGpuEnabled) {
		MatrixUtil.isGpuEnabled = isGpuEnabled;
	}

	static {
		seed = System.currentTimeMillis();
		random = new Random(seed);
	}

	/**
	 * Sets the seed of the pseudo-random number generator. This method enables
	 * you to produce the same sequence of "random" number for each execution of
	 * the program. Ordinarily, you should call this method at most once per
	 * program.
	 *
	 * @param s
	 *            the seed
	 */
	public static void setSeed(long s) {
		seed = s;
		random = new Random(seed);
	}

	/**
	 * Returns the seed of the pseudo-random number generator.
	 *
	 * @return the seed
	 */
	public static long getSeed() {
		return seed;
	}

	/**
	 * Returns a random real number uniformly in [0, 1).
	 *
	 * @return a random real number uniformly in [0, 1)
	 */
	public static double uniform() {
		return random.nextDouble();
	}

	public static float uniformF() {
		return random.nextFloat();
	}

	public static double normal() {
		return random.nextDouble();
	}

	/**
	 * Returns a random integer uniformly in [0, n).
	 *
	 * @param n
	 *            number of possible integers
	 * @return a random integer uniformly between 0 (inclusive) and {@code n}
	 *         (exclusive)
	 * @throws IllegalArgumentException
	 *             if {@code n <= 0}
	 */
	public static int uniform(int n) {
		if (n <= 0) {
			throw new IllegalArgumentException("argument must be positive: "
					+ n);
		}
		return random.nextInt(n);
	}

	/**
	 * Returns a random long integer uniformly in [0, n).
	 *
	 * @param n
	 *            number of possible {@code long} integers
	 * @return a random long integer uniformly between 0 (inclusive) and
	 *         {@code n} (exclusive)
	 * @throws IllegalArgumentException
	 *             if {@code n <= 0}
	 */
	public static long uniform(long n) {
		if (n <= 0L) {
			throw new IllegalArgumentException("argument must be positive: "
					+ n);
		}

		long r = random.nextLong();
		long m = n - 1;

		// power of two
		if ((n & m) == 0L) {
			return r & m;
		}

		// reject over-represented candidates
		long u = r >>> 1;
		while (u + m - (r = u % n) < 0L) {
			u = random.nextLong() >>> 1;
		}
		return r;
	}

	/**
	 * Returns a random integer uniformly in [a, b).
	 *
	 * @param a
	 *            the left endpoint
	 * @param b
	 *            the right endpoint
	 * @return a random integer uniformly in [a, b)
	 * @throws IllegalArgumentException
	 *             if {@code b <= a}
	 * @throws IllegalArgumentException
	 *             if {@code b - a >= Integer.MAX_VALUE}
	 */
	public static int uniform(int a, int b) {
		if ((b <= a) || ((long) b - a >= Integer.MAX_VALUE)) {
			throw new IllegalArgumentException("invalid range: [" + a + ", "
					+ b + ")");
		}
		return a + uniform(b - a);
	}

	/**
	 * Returns a random real number uniformly in [a, b).
	 *
	 * @param a
	 *            the left endpoint
	 * @param b
	 *            the right endpoint
	 * @return a random real number uniformly in [a, b)
	 * @throws IllegalArgumentException
	 *             unless {@code a < b}
	 */
	public static double uniform(double a, double b) {
		if (!(a < b)) {
			throw new IllegalArgumentException("invalid range: [" + a + ", "
					+ b + ")");
		}
		return a + uniform() * (b - a);
	}

	public static float uniformF(float a, float b) {
		if (!(a < b)) {
			throw new IllegalArgumentException("invalid range: [" + a + ", "
					+ b + ")");
		}
		return a + uniformF() * (b - a);
	}

	public static double normal(double a, double b) {
		if (!(a < b)) {
			throw new IllegalArgumentException("invalid range: [" + a + ", "
					+ b + ")");
		}
		return normal();
	}

	/**
	 * @param m
	 * @param n
	 * @return random m-by-n matrix with values between 0 and 1
	 * @throws IllegalAccessException
	 */
	public static double[][] random(int m, int n) throws IllegalAccessException {
		if (isGpuEnabled) {
			return MatrixUtilGPU.generateRandom(m, n,
					ProbablityDistributionTypes.UNIFORM);
		} else {
			double[][] a = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					a[i][j] = uniform(0.0, 1.0);
				}
			}
			return a;
		}
	}
/**
 * 
 * @param m
 * @param n
 * @return
 * @throws IllegalAccessException
 */
	public static float[][] randomF(int m, int n) throws IllegalAccessException {

		float[][] a = new float[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				a[i][j] = uniformF(0.0f, 1.0f);
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
 * @throws IllegalAccessException
 */
	public static double[][] random(int m, int n,
			ProbablityDistributionTypes probType) throws IllegalAccessException {
		if (isGpuEnabled) {
			return MatrixUtilGPU.generateRandom(m, n,
					ProbablityDistributionTypes.UNIFORM);
		} else {

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
 * 
 * @param arr
 */
	static void T_ref(int arr[][]) {
		for (int i = 0; i < arr.length; i++)
			for (int j = i; j < arr[0].length; j++) {
				int temp = arr[j][i];
				arr[j][i] = arr[i][j];
				arr[i][j] = temp;
			}
	}

	/**
	 * get log of all matrix
	 * 
	 * @param a
	 * @return
	 * @throws Exception
	 */
	public static double[][] log(double[][] a) throws Exception {

		if (isGpuEnabled) {
			return MatrixUtilGPU.simplePointWiseFunctions(a, StandardMathFunctions.LOG);
		} else {

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
	}
/**
 * 
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] sqrt(double[][] a) throws Exception {
		if (isGpuEnabled) {
			return MatrixUtilGPU
					.simplePointWiseFunctions(a, StandardMathFunctions.SQRT);
		} else {

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
		// np.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;
		if (isGpuEnabled) {
			return MatrixUtilGPU.add(a, b);
		} else {
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					c[i][j] = a[i][j] + b[i][j];

				}
			}
			return c;
		}

	}
/**
 * 
 * @param a
 * @param b
 * @return
 * @throws Exception
 */
	public static float[][] add(float[][] a, float[][] b) throws Exception {
		// np.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;
		if (isGpuEnabled) {
			return MatrixUtilGPU.add(a, b);
		} else {
			float[][] c = new float[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					c[i][j] = a[i][j] + b[i][j];

				}
			}
			return c;
		}

	}
/**
 * Split into train and test set according to ratio
 * @param X
 * @param Y
 * @param trainRatio
 * @return
 * @throws IllegalAccessException
 */
	public static Map<Enum<DeepNNKeysForMaps>, Object> splitIntoTrainAndTestForRegression(
			double[][] X, double[][] Y, double trainRatio)
			throws IllegalAccessException {
		// np.printshapes(a,b);
		int m = X.length;
		int n = X[0].length;
		Map<Enum<DeepNNKeysForMaps>, Object> resultMap = new HashMap<Enum<DeepNNKeysForMaps>, Object>();
		int lenghtOfTrainSet = (int) (((double) X.length) * trainRatio);
		int lenghtOfTestSet = m - lenghtOfTrainSet;
		double[][] randomVector = MatrixUtil.random(m, 1);
		double[][] X_trainSet = new double[lenghtOfTrainSet][n];
		double[][] X_testSet = new double[lenghtOfTestSet][n];
		double[][] Y_trainSet = new double[lenghtOfTrainSet][1];
		double[][] Y_testSet = new double[lenghtOfTestSet][1];
		int minimumset = Math.min(lenghtOfTrainSet, lenghtOfTestSet);
		int maximummset = Math.min(lenghtOfTrainSet, lenghtOfTestSet);
		int timesnoConditionMet = 0;
		for (int i = 0; i < m; i++) {

			if ((randomVector[i][0] > 0.5) && (lenghtOfTestSet > 0)) {
				lenghtOfTestSet--;
				// System.out.println(lenghtOfTestSet);
				X_testSet[lenghtOfTestSet] = X[i];
				Y_testSet[lenghtOfTestSet] = Y[i];
				minimumset--;
			} else if ((randomVector[i][0] <= 0.5) && lenghtOfTrainSet > 0) {
				lenghtOfTrainSet--;
				X_trainSet[lenghtOfTrainSet] = X[i];
				Y_trainSet[lenghtOfTrainSet] = Y[i];
				maximummset--;
			} else if (lenghtOfTrainSet > 0) {
				lenghtOfTrainSet--;
				X_trainSet[lenghtOfTrainSet] = X[i];
				Y_trainSet[lenghtOfTrainSet] = Y[i];
				maximummset--;
			} else if (lenghtOfTestSet > 0) {
				lenghtOfTestSet--;
				X_testSet[lenghtOfTestSet] = X[i];
				Y_testSet[lenghtOfTestSet] = Y[i];
				minimumset--;
			} else {
				timesnoConditionMet++;
			}

		}
		
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX, X_trainSet);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL, Y_trainSet);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX_TEST, X_testSet);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL_TEST, Y_testSet);
		return resultMap;
	}
/**
 * Splitting into train test part is not easy task for multiclass classification task. I know it is not perfect. If you have any idea please let me know.
 * Becaue every mini batch should have examples from all the classes. What if we have skewed classes.?
 * @param X
 * @param Y
 * @param trainRatio
 * @return
 */
	public static Map<Enum<DeepNNKeysForMaps>, Object> splitIntoTrainAndTestForClassification(
			double[][] X, double[][] Y, double trainRatio) {
		// np.printshapes(a,b);
		int m = X.length;
		int n = X[0].length;
		Map<Enum<DeepNNKeysForMaps>, Object> resultMap = new HashMap<Enum<DeepNNKeysForMaps>, Object>();
		int lenghtOfTrainSet = (int) (((double) X.length) * trainRatio);
		int lenghtOfTestSet = m - lenghtOfTrainSet;

		double[][] X_trainSet = new double[lenghtOfTrainSet][n];
		double[][] X_testSet = new double[lenghtOfTestSet][n];
		double[][] Y_trainSet = new double[lenghtOfTrainSet][1];
		double[][] Y_testSet = new double[lenghtOfTestSet][1];

		Map<Double, Integer> labelsCountMap = new HashMap<>();
		int individualCount = 0;
		for (int i = 0; i < Y.length; i++) {

			if (!labelsCountMap.containsKey(Y[i][0])) {
				individualCount = 1;
				labelsCountMap.put(Y[i][0], individualCount);
			} else {
				individualCount = labelsCountMap.get(Y[i][0]) + 1;
				labelsCountMap.put(Y[i][0], individualCount);
			}

		}
		List<Double> tmplist = new ArrayList<Double>(labelsCountMap.keySet());
		Collections.sort(tmplist);
		System.out.println("total number of classes identified: "
				+ labelsCountMap.keySet().size() + ": "
				+ Arrays.toString(tmplist.toArray()));
		Set<Integer> setOfAllTestIndeces = new HashSet();
		for (double label : labelsCountMap.keySet()) {
			int bumberOfitemsForSingleLabel = labelsCountMap.get(label);
			int lenghtOfTrainSetForSingleLable = (int) (((double) bumberOfitemsForSingleLabel) * trainRatio);
			int lenghtOfTestSetForSingleItem = bumberOfitemsForSingleLabel
					- lenghtOfTrainSetForSingleLable;
			Integer[] uniqueLabelIndeces = MatrixUtil
					.getItemIndexVectForMultiClassFileData(Y, label);
			MatrixUtil.selectNRandomNumbersFromVector(uniqueLabelIndeces,
					lenghtOfTestSetForSingleItem, setOfAllTestIndeces);
		}
		for (int i = 0; i < Y.length; i++) {
			if (setOfAllTestIndeces.contains(i) && lenghtOfTestSet > 0) {
				lenghtOfTestSet--;
				X_testSet[lenghtOfTestSet] = X[i];
				Y_testSet[lenghtOfTestSet] = Y[i];

			} else if (lenghtOfTrainSet > 0) {
				lenghtOfTrainSet--;
				X_trainSet[lenghtOfTrainSet] = X[i];
				Y_trainSet[lenghtOfTrainSet] = Y[i];

			} else if (lenghtOfTestSet > 0) {
				lenghtOfTestSet--;
				X_testSet[lenghtOfTestSet] = X[i];
				Y_testSet[lenghtOfTestSet] = Y[i];

			}
		}

		resultMap.put(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX, X_trainSet);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL, Y_trainSet);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX_TEST, X_testSet);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL_TEST, Y_testSet);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_NUMBER_OF_CLASSES,
				labelsCountMap.keySet().size());
		return resultMap;
	}
/**
 * 
 * @param uniqueLabelIndeces
 * @param lenghtOfTestSetForSingleItem
 * @param setOfAllTestIndeces
 */
	private static void selectNRandomNumbersFromVector(
			Integer[] uniqueLabelIndeces, int lenghtOfTestSetForSingleItem,
			Set<Integer> setOfAllTestIndeces) {
		// TODO Auto-generated method stub
		// Integer[] randomValues=new Integer[lenghtOfTestSetForSingleItem];
		int i = 0;
		while (i <= lenghtOfTestSetForSingleItem) {
			int randomIndex = MatrixUtil.random.nextInt(uniqueLabelIndeces.length) + 0;
			setOfAllTestIndeces.add(uniqueLabelIndeces[randomIndex]);
			i++;
		}
		// return randomValues;
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
	 * This is python numpy version of np.add where the b is rescalled to fector
	 * or matrix on the basis of A
	 * 
	 * @param a
	 * @param b
	 * @param isrescale_b
	 * @return
	 * @throws Exception
	 */
	public static double[][] subtract(double[][] a, double[][] b,
			boolean isrescale_b) throws Exception {
		// np.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;
		if (isGpuEnabled) {
			// System.out.println("GPU is called for multiplication");
			return MatrixUtilGPU.subtract(a, b, isrescale_b);
		} else {
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					if (isrescale_b) {
						if (b[0].length == 1 && b.length == m) {
							c[i][j] = a[i][j] - b[i][0];
						} else if ((b[0].length == 1) && b.length == 1) {
							c[i][j] = a[i][j] - b[0][0];
						} else if (b[0].length == n && b.length == 1) {

							c[i][j] = a[i][j] - b[0][j];

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

	}
/**
 * 
 * @param a
 * @param b
 * @param axis
 * @return
 * @throws Exception
 */
	public static double[][] subtractVector(double[][] a, double[] b, int axis)
			throws Exception {
		// np.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;

		double[][] c = null;

		if (axis == 1) {
			c = new double[m][n];
			;
			for (int i = 0; i < m; i++) {

				for (int j = 0; j < n; j++) {

					c[i][j] = a[i][j] - b[i];

				}

			}
		} else if (axis == 0) {
			c = new double[n][m];
			for (int j = 0; j < n; j++) {

				for (int i = 0; i < m; i++) {

					c[i][j] = a[i][j] - b[i];
					;

				}

			}
		}
		return c;

	}
/**
 * 
 * @param a
 * @param b
 * @param axis
 * @return
 * @throws Exception
 */
	public static double[][] divideVector(double[][] a, double[] b, int axis)
			throws Exception {
		// np.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;

		double[][] c = null;
		if (axis == 1) {
			c = new double[m][n];
			;
			for (int i = 0; i < m; i++) {

				for (int j = 0; j < n; j++) {

					if( b[i]!=0){
						c[i][j] = a[i][j] / b[i];	
					}

				}

			}
		} else if (axis == 0) {
			c = new double[n][m];
			for (int j = 0; j < n; j++) {

				for (int i = 0; i < m; i++) {
					if( b[i]!=0){
						c[i][j] = a[i][j] / b[i];	
					}
				
					;

				}

			}
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
	public static double[][] add(double[][] a, double[][] b, boolean isrescale_b)
			throws Exception {
		// np.printshapes(a,b);
		int m = a.length;
		int n = a[0].length;
		if (isGpuEnabled) {
			// System.out.println("GPU is called for multiplication");
			return MatrixUtilGPU.add(a, b, isrescale_b);
		} else {
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					if (isrescale_b) {
						if (b[0].length == 1 && b.length == m) {
							c[i][j] = a[i][j] + b[i][0];
						} else if ((b[0].length == 1) && b.length == 1) {
							c[i][j] = a[i][j] + b[0][0];
						}

						else {
							c[i][j] = a[i][j] + b[i][j];
						}

					} else {
						c[i][j] = a[i][j] + b[i][j];
					}

				}
			}
			return c;
		}

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
 */
	public static double[][] add(double[][] a, double b) {
		int m = a.length;
		int n = a[0].length;
		double[][] c = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				c[i][j] = a[i][j] + b;
			}
		}
		return c;
	}

	/**
	 * @param a
	 *            matrix
	 * @param b
	 *            matrix
	 * @return c = a - b
	 */
	public static double[][] subtract(double[][] a, double[][] b) {
		int m = a.length;
		int n = a[0].length;
		double[][] c = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				c[i][j] = a[i][j] - b[i][j];
			}
		}
		return c;
	}
/**
 * 
 * @param matrix
 */
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

		if (isGpuEnabled) {
			return MatrixUtilGPU.subtract(a, b);
		} else {
			int m = b.length;
			int n = b[0].length;
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					c[i][j] = a - b[i][j];
				}
			}
			return c;
		}
	}
/**
 * 
 * @param b
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] subtract(double[][] b, double a) throws Exception {
		if (isGpuEnabled) {
			return MatrixUtilGPU.subtract(b, a);
		} else {

			int m = b.length;
			int n = b[0].length;
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					c[i][j] = b[i][j] - a;
				}
			}
			return c;
		}
	}

	/**
	 * @param a
	 *            matrix
	 * @param b
	 *            matrix
	 * @return c = a * b
	 * @throws Exception
	 */
	public static double[][] dot(double[][] a, double[][] b) throws Exception {
		if (isGpuEnabled) {
			// Stopwatch stopWatch = new Stopwatch();
			double[][] y = MatrixUtilGPU.dot(a, b);
			/*
			 * double gpuTime=stopWatch.elapsedTime();
			 * 
			 * System.out.println(" time taken in multiplication when on gpu: "
			 * + gpuTime + " seconds \n");
			 */

			return y;
		} else {
			// Stopwatch stopWatch = new Stopwatch();
			int m1 = a.length;
			int n1 = a[0].length;
			int m2 = b.length;
			int n2 = b[0].length;
			if (n1 != m2) {
				throw new RuntimeException("Illegal matrix dimensions.");
			}
			// System.out.println("started :multiplication elements "+(m1*n1));
			double[][] c = new double[m1][n2];
			for (int i = 0; i < m1; i++) {
				for (int j = 0; j < n2; j++) {
					for (int k = 0; k < n1; k++) {

						c[i][j] += a[i][k] * b[k][j];

					}
				}
			}
			/*
			 * System.out.println("Done :multiplication "+(m1*n1));
			 * 
			 * double gpuTime=stopWatch.elapsedTime();
			 * 
			 * System.out.println(" time taken in multiplication on CPU: " +
			 * gpuTime
			 * 
			 * + " seconds \n");
			 */
			return c;
		}
	}
/**
 * Equality check of two LargeDoubleMatrix 
 * @param a
 * @param b
 * @return
 */
	public static boolean equalityCheck(LargeDoubleMatrix a, LargeDoubleMatrix b) {
		boolean res = true;
		int m1 = a.width();
		int n1 = a.height();
		int m2 = b.width();
		int n2 = b.height();
		if (a.width() != b.width() || a.height() != b.height()) {
			return false;
		}
		for (int i = 0; i < m1; i++) {
			for (int j = 0; j < n2; j++) {
				if (Double.compare(a.get(i, j), b.get(i, j)) != 0) {
					return false;
				}
			}
		}
		return res;
	}
/**
 * 
 * @param a
 * @param b
 * @return
 */
	public static boolean equalityCheck(double[][] a, double[][] b) {
		boolean res = true;
		int m1 = a.length;
		int n1 = a[0].length;
		int m2 = b.length;
		int n2 = b[0].length;
		if (m1 != m2 || n1 != n2) {
			return false;
		}
		for (int i = 0; i < m1; i++) {
			for (int j = 0; j < n2; j++) {
				if (Double.compare(a[i][j], b[i][j]) != 0) {
					System.out.println(a[i][j] - b[i][j]);
					return false;
				}
			}
		}
		return res;
	}
/**
 * 
 * @param a
 * @param b
 * @param resultMatr
 * @throws IOException
 * @throws InterruptedException
 */
	public static void dot(LargeDoubleMatrix a, LargeDoubleMatrix b,
			LargeDoubleMatrix resultMatr) throws IOException,
			InterruptedException {

		if (isGpuEnabled) {
			Stopwatch stopWatch = new Stopwatch();

			MatrixUtilGPU.dot(a, b, resultMatr);

			double gpuTime = stopWatch.elapsedTime();

			System.out.println(" time taken in multiplication when on gpu: "
					+ gpuTime + " seconds \n");

		} else {
			int m1 = a.height();
			int n1 = a.width();
			int m2 = b.height();
			int n2 = b.width();
			double sum = 0.0;
			int c = 0;
			if (n1 != m2) {
				throw new RuntimeException("Illegal matrix dimensions.");
			}

			for (int i = 0; i < m1; i++) {

				for (int j = 0; j < n2; j++) {

					for (int k = 0; k < n1; k++) {

						sum = resultMatr.get(i, j);
						sum = sum + (a.get(i, k) * b.get(k, j));
						resultMatr.set(i, j, sum);

					}

				}
			}
		}
		// return resultMatr;

	}

	/**
	 * @param a
	 *            matrix
	 * @param b
	 *            matrix
	 * @return c = a * b
	 * @throws Exception
	 */
	public static float[][] dot(float[][] a, float[][] b) throws Exception {
		if (isGpuEnabled) {
			// Stopwatch stopWatch = new Stopwatch();
			float[][] y = MatrixUtilGPU.dot(a, b);
			/*
			 * double gpuTime=stopWatch.elapsedTime();
			 * 
			 * System.out.println(" time taken in multiplication when on gpu: "
			 * + gpuTime
			 * 
			 * + " seconds \n");
			 */
			return y;
		} else {
			Stopwatch stopWatch = new Stopwatch();
			int m1 = a.length;
			int n1 = a[0].length;
			int m2 = b.length;
			int n2 = b[0].length;
			if (n1 != m2) {
				throw new RuntimeException("Illegal matrix dimensions.");
			}
			// System.out.println("started :multiplication elements "+(m1*n1));
			float[][] c = new float[m1][n2];
			for (int i = 0; i < m1; i++) {
				for (int j = 0; j < n2; j++) {
					for (int k = 0; k < n1; k++) {

						c[i][j] += a[i][k] * b[k][j];

					}
				}
			}
			/*
			 * System.out.println("Done :multiplication "+(m1*n1));
			 * 
			 * double gpuTime=stopWatch.elapsedTime();
			 * 
			 * System.out.println(" time taken in multiplication on CPU: " +
			 * gpuTime
			 * 
			 * + " seconds \n");
			 */
			return c;
		}
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
 * 
 * @param a
 * @param axis
 * @return
 */
	public static float[][] sum(float[][] a, int axis) {
		int m1 = a.length;
		int n1 = a[0].length;
		float[][] c = null;
		if (axis == 1) {
			c = new float[m1][1];
			for (int i = 0; i < m1; i++) {
				float sum = 0;
				for (int j = 0; j < n1; j++) {

					sum = sum + a[i][j];

				}
				c[i][0] = sum;
			}
		} else if (axis == 0) {
			c = new float[1][n1];
			for (int j = 0; j < n1; j++) {
				float sum = 0;
				for (int i = 0; i < m1; i++) {

					sum = sum + a[i][j];

				}
				c[0][j] = sum;
			}
		}

		return c;
	}
/**
 * 
 * @param a
 * @return
 */
	public static float[][] diag(float[][] a) {
		int size = a.length;
		boolean colWise = false;
		if (size == 1) {
			size = a[0].length;
			colWise = true;
		}
		float[][] dig = new float[size][size];
		for (int i = 0; i < size; i++) {
			if (colWise) {
				dig[i][i] = a[0][i];
			} else {
				dig[i][i] = a[i][0];
			}

		}
		return dig;
	}
/**
 * 
 * @param a
 * @return
 */
	public static double[][] diag(double[][] a) {
		int size = a.length;
		boolean colWise = false;
		if (size == 1) {
			size = a[0].length;
			colWise = true;
		}
		double[][] dig = new double[size][size];
		for (int i = 0; i < size; i++) {
			if (colWise) {
				dig[i][i] = a[0][i];
			} else {
				dig[i][i] = a[i][0];
			}

		}
		return dig;
	}
/**
 * 
 * @param a
 * @param axis
 * @return
 */
	public static int[][] sum(int[][] a, int axis) {
		int m1 = a.length;
		int n1 = a[0].length;
		int[][] c = null;
		if (axis == 1) {
			c = new int[m1][1];
			for (int i = 0; i < m1; i++) {
				int sum = 0;
				for (int j = 0; j < n1; j++) {

					sum = sum + a[i][j];

				}
				c[i][0] = sum;
			}
		} else if (axis == 0) {
			c = new int[1][n1];
			for (int j = 0; j < n1; j++) {
				int sum = 0;
				for (int i = 0; i < m1; i++) {

					sum = sum + a[i][j];

				}
				c[0][j] = sum;
			}
		}

		return c;
	}
/**
 * 
 * @param x
 * @return
 */
	public static double sum(double[] x) {
		int m = x.length;
		double sum = 0;

		for (int j = 0; j < m; j++) {
			sum = sum + x[j];
		}
		return sum;
	}
/**
 * 
 * @param x
 * @return
 */
	public static double sum(float[] x) {
		int m = x.length;
		float sum = 0;

		for (int j = 0; j < m; j++) {
			sum = sum + x[j];
		}
		return sum;
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
/**
 * 
 * @param squareMatrix
 * @return
 */
	public static double[][] addIdentityToAMatrix(double[][] squareMatrix) {
		int m = squareMatrix.length;
		for (int j = 0; j < m; j++) {
			squareMatrix[j][j] = 1.0 + squareMatrix[j][j];
		}
		return squareMatrix;
	}
/**
 * 
 * @param x
 * @param label
 * @return
 */
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
/**
 * 
 * @param x
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] multiply(double[][] x, double[][] a)
			throws Exception {
		int m = a.length;
		int n = a[0].length;
		double gpuTime = 0;
		double cpuTime = 0;
		if (x.length != m || x[0].length != n) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		if (isGpuEnabled) {
			// Stopwatch stopWatch = new Stopwatch();
			double[][] y = MatrixUtilGPU.multiply(x, a);
			/*
			 * gpuTime=stopWatch.elapsedTime();
			 * 
			 * System.out.println(" time taken in multiplication when gpu is on: "
			 * + gpuTime
			 * 
			 * + " seconds \n");
			 */
			return y;
		} else {
			Stopwatch stopWatch = new Stopwatch();
			double[][] y = new double[m][n];
			for (int j = 0; j < m; j++) {
				for (int i = 0; i < n; i++) {
					y[j][i] = a[j][i] * x[j][i];
				}
			}
			/*
			 * cpuTime=stopWatch.elapsedTime();
			 * System.out.println(" time taken in multiplication when gpu is off: "
			 * + cpuTime
			 * 
			 * + " seconds \n"); System.out.println("percentage increment : " +
			 * ((cpuTime-gpuTime)/gpuTime)*100+" % ");
			 */
			return y;
		}

	}
/**
 * 
 * @param x
 * @param a
 * @return
 * @throws Exception
 */
	public static float[][] multiply(float[][] x, float[][] a) throws Exception {
		int m = a.length;
		int n = a[0].length;
		double gpuTime = 0;
		double cpuTime = 0;
		if (x.length != m || x[0].length != n) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		if (isGpuEnabled) {
			// Stopwatch stopWatch = new Stopwatch();
			float[][] y = MatrixUtilGPU.multiply(x, a);

			return y;
		} else {
			Stopwatch stopWatch = new Stopwatch();
			float[][] y = new float[m][n];
			for (int j = 0; j < m; j++) {
				for (int i = 0; i < n; i++) {
					y[j][i] = a[j][i] * x[j][i];
				}
			}

			return y;
		}

	}
/**
 * 
 * @param a
 * @param x
 * @return
 * @throws Exception
 */
	public static double[][] multiply(double[][] a, double x) throws Exception {
		int m = a.length;
		int n = a[0].length;

		if (isGpuEnabled) {
			return MatrixUtilGPU.multiply(x, a);
		} else {
			double[][] y = new double[m][n];
			for (int j = 0; j < m; j++) {
				for (int i = 0; i < n; i++) {
					y[j][i] = a[j][i] * x;
				}
			}
			return y;
		}
	}

	/**
	 * Element wise multiplication
	 *
	 * @param a
	 *            matrix
	 * @param x
	 *            scaler
	 * @return y = a * x
	 * @throws Exception
	 */
	public static double[][] multiply(double x, double[][] a) throws Exception {
		int m = a.length;
		int n = a[0].length;

		if (isGpuEnabled) {
			return MatrixUtilGPU.multiply(x, a);
		} else {
			double[][] y = new double[m][n];
			for (int j = 0; j < m; j++) {
				for (int i = 0; i < n; i++) {
					y[j][i] = a[j][i] * x;
				}
			}
			return y;
		}
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

		/*
		 * if(isGpuEnabled) { return npGpu.power(x, a); } else
		 */
		{
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

	}
/**
 * 
 * @param x
 * @param a
 * @return
 */
	public static double[][] power(double[][] x, double a) {

		/*
		 * if(isGpuEnabled) { return npGpu.power(x, a); } else
		 */
		{
			int m = x.length;
			int n = x[0].length;

			double[][] y = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					if (x[i][j] != 0.0) {
						if (a < 0) {
							y[i][j] = 1.0 / Math.pow(x[i][j], -a);
						} else {
							y[i][j] = Math.pow(x[i][j], a);
						}

					}

				}
			}
			return y;
		}

	}
/**
 * 
 * @param x
 * @param a
 * @return
 * @throws Exception
 */
	public static float[][] power(float[][] x, float a) throws Exception {

		/*
		 * if(isGpuEnabled) { return npGpu.power(x, a); } else
		 */
		{
			int m = x.length;
			int n = x[0].length;
			float tmp = 0.0f;

			float[][] y = new float[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					if (x[i][j] == 0.0f) {
						tmp = 0;
					} else {
						tmp = (float) Math.pow(x[i][j], a);
					}

					if (Float.isNaN(tmp) || Float.isInfinite(tmp)) {
						throw new Exception(" not valid power" + x[i][j]);
					} else {
						y[i][j] = (float) Math.pow(x[i][j], a);
					}

				}
			}
			return y;
		}

	}

	/**
	 * 
	 * @param x
	 * @return
	 * @throws Exception
	 */
	public static double[][] exp(double[][] x) throws Exception {
		double gpuTime = 0;
		double cpuTime = 0;

		if (isGpuEnabled) {
			Stopwatch stopWatch = new Stopwatch();
			double[][] y = MatrixUtilGPU.simplePointWiseFunctions(x,
					StandardMathFunctions.EXP);
			gpuTime = stopWatch.elapsedTime();
			System.out.println(" time taken EXP gpu is on: " + gpuTime

			+ " seconds \n");
			return MatrixUtilGPU.simplePointWiseFunctions(x, StandardMathFunctions.EXP);
		} else {
			// Stopwatch stopWatch = new Stopwatch();
			int m = x.length;
			int n = x[0].length;

			double[][] y = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					y[i][j] = Math.exp(x[i][j]);
				}
			}
			/*
			 * cpuTime=stopWatch.elapsedTime();
			 * System.out.println(" time taken EXP gpu is off: " + cpuTime
			 * 
			 * + " seconds \n"); System.out.println("percentage increment exp: "
			 * + ((cpuTime-gpuTime)/gpuTime)*100+" % ");
			 */

			return y;
		}
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
		// print(Vshape);
		return Vshape;
	}
/**
 * 
 * @param a
 * @return
 */
	public static double[] shapeDim(double[][] a) {
		int m = a.length;
		int n = a[0].length;

		return new double[] { m, n };
	}
/**
 * 
 * @param a
 * @return
 */
	public static String shape(float[][] a) {
		int m = a.length;
		int n = a[0].length;
		String Vshape = "(" + m + "," + n + ")";
		return Vshape;
	}
/**
 * 
 * @param a
 * @return
 */
	public static double[] shapeDim(float[][] a) {
		int m = a.length;
		int n = a[0].length;

		return new double[] { m, n };
	}
/**
 * 
 * @param nums
 * @param r
 * @param c
 * @return
 */
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
/**
 * 
 * @param nums
 * @param r
 * @param c
 * @return
 */
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
/**
 * 
 * @param src
 * @return
 */
	public static double[][] cloneArray(double[][] src) {
		int length = src.length;
		double[][] target = new double[length][src[0].length];
		for (int i = 0; i < length; i++) {
			System.arraycopy(src[i], 0, target[i], 0, src[i].length);
		}
		return target;
	}
/**
 * 
 * @param a
 * @param other
 * @param valueToBeReplaced
 * @return
 */
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
/**
 * 
 * @param a
 * @param other
 * @param valueToBeReplaced
 * @return
 */
	public static float[][] replaceNegativeValuesOnOtherArrayBasis(float[][] a,
			float[][] other, float valueToBeReplaced) {
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
/**
 * 
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] normalizedAdjSpectralMethod(double[][] a)
			throws Exception {
		int m = a.length;
		int n = a[0].length;

		double[][] identityMatrix = MatrixUtil.createIdentity(m, m);
		double[][] A_hat = MatrixUtil.add(a, identityMatrix); // avoiding self loop
		double[][] rowSum = MatrixUtil.sum(a, 1);
		double[][] rowSumInvSqrt = MatrixUtil.power(rowSum, -0.5);
		double[][] digInvSqrt = MatrixUtil.diag(rowSumInvSqrt);
		return MatrixUtil.dot(MatrixUtil.T(MatrixUtil.dot(A_hat, digInvSqrt)), digInvSqrt);
	}
/**
 * 
 * @param a
 * @return
 * @throws Exception
 */
	public static float[][] normalizedAdjSpectralMethod(float[][] a)
			throws Exception {
		int m = a.length;
		int n = a[0].length;

		float[][] identityMatrix = MatrixUtil.createIdentityFloat(m, m);
		float[][] A_hat = MatrixUtil.add(a, identityMatrix); // avoiding self loop
		float[][] rowSum = MatrixUtil.sum(a, 1);
		float[][] rowSumInvSqrt = MatrixUtil.power(rowSum, -0.5f);
		float[][] digInvSqrt = MatrixUtil.diag(rowSumInvSqrt);
		return MatrixUtil.dot(MatrixUtil.T(MatrixUtil.dot(A_hat, digInvSqrt)), digInvSqrt);
	}
/**
 * 
 * @param a
 * @param threshold
 * @return
 * @throws Exception
 */
	public static double[][] createBooleanMatrixForSomeThreshold(double[][] a,
			double threshold) throws Exception {
		if (isGpuEnabled) {
			return MatrixUtilGPU.createBooleanMatrixForSomeThreshold(a, threshold);
		} else {
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
	}
/**
 * 
 * @param m
 * @param n
 * @param constantValue
 * @return
 */
	public static double[][] createMatrixWithConstantValue(int m, int n,
			double constantValue) {

		double[][] a = new double[m][n];
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++) {
				a[i][j] = constantValue;
			}
		return a;
	}
/**
 * 
 * @param m
 * @param n
 * @return
 */
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
 * 
 * @param m
 * @param n
 * @return
 */
	public static float[][] createIdentityFloat(int m, int n) {

		float[][] a = new float[m][n];
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++) {
				if (i == j) {
					a[i][j] = 1;
				} else {
					a[i][j] = 0;
				}

			}
		return a;
	}

	/**
	 * @param a
	 *            matrix
	 * @return sigmoid of matrix a
	 * @throws Exception
	 */
	public static double[][] sigmoid(double[][] a) throws Exception {

		if (isGpuEnabled) {
			return MatrixUtilGPU.activations(a, DeepNNActivations.SIGMOID);
		} else {

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
	}
/**
 * Swish activatoin
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] swish(double[][] a) throws Exception {
		/*
		 * if(isGpuEnabled) { return npGpu.activations(a,
		 * DeepNNActivations.SWISH); } else
		 */

		{
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
	}
/**
 * Tanh activation
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] tanh(double[][] a) throws Exception {

		if (isGpuEnabled) {
			return MatrixUtilGPU.tanh(a);
		}

		else

		{
			int m = a.length;
			int n = a[0].length;
			double[][] z = new double[m][n];
			double tmpExpPositve = 0.0, tmpExpNegative = 0.0;
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					tmpExpPositve = Math.exp(a[i][j]);
					tmpExpNegative = Math.exp(-a[i][j]);
					z[i][j] = ((tmpExpPositve - tmpExpNegative) / (tmpExpPositve + tmpExpNegative));
				}
			}
			return z;
		}

	}
/**
 * This method is for testing how to add your own activation. This activation will be applied during forward operations. 
 * 
 * @param a
 * @return
 */
	public static double[][] ktest(double[][] a) {
		int m = a.length;
		int n = a[0].length;
		double[][] z = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				
				z[i][j] = ((Math.exp(a[i][j])) / (Math.exp(a[i][j]) + Math
						.exp(-a[i][j])));
			}
		}
		return z;
	}
/**
 * Relu activatoin
 * @param a
 * @return
 */
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

	/**
	 * Element wise division
	 *
	 * @param a
	 *            scaler
	 * @param x
	 *            matrix
	 * @return x / a
	 */
	public static double[][] divide(double[][] x, int a) {
		int m = x.length;
		int n = x[0].length;

		double[][] z = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				z[i][j] = (x[i][j] / a);
			}
		}
		return z;
	}
/**
 * 
 * @param x
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] divide(double[][] x, double a) throws Exception {
		if (isGpuEnabled) {
			return MatrixUtilGPU.divide(x, a);
		} else {
			int m = x.length;
			int n = x[0].length;

			double[][] z = new double[m][n];

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					z[i][j] = (x[i][j] / a);
				}
			}
			return z;
		}
	}
/**
 * 
 * @param x
 * @param a
 * @return
 * @throws Exception
 */
	public static float[][] divide(float[][] x, float a) throws Exception {
		if (isGpuEnabled) {
			return MatrixUtilGPU.divide(x, a);
		} else {
			int m = x.length;
			int n = x[0].length;

			float[][] z = new float[m][n];

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					z[i][j] = (x[i][j] / a);
				}
			}
			return z;
		}
	}
/**
 * Element wise division where second operant is a scalar.
 * @param x
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] divide(int[][] x, double a) throws Exception {
		/*
		 * if(isGpuEnabled) { return npGpu.divide(x, a); } else
		 */

		{
			int m = x.length;
			int n = x[0].length;

			double[][] z = new double[m][n];

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					z[i][j] = (x[i][j] / a);
				}
			}
			return z;
		}
	}
/**
 * Elementwise division where first operantis a scalar.
 * @param a
 * @param x
 * @return
 * @throws Exception
 */
	public static double[][] divide(double a, double[][] x) throws Exception {

		if (isGpuEnabled) {
			return MatrixUtilGPU.divide(a, x);
		} else {

			int m = x.length;
			int n = x[0].length;

			double[][] z = new double[m][n];

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					z[i][j] = (a / x[i][j]);
				}
			}
			return z;
		}
	}
/**
 * Elementwise division of two matrices
 * @param x
 * @param a
 * @return
 * @throws Exception
 */
	public static double[][] divide(double[][] x, double[][] a)
			throws Exception {

		if (isGpuEnabled) {
			return MatrixUtilGPU.divide(x, a);
		} else {

			int m = x.length;
			int n = x[0].length;

			double[][] z = new double[m][n];

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					if (a[i][j] != 0) {
						z[i][j] = (x[i][j] / a[i][j]);
					} else {
						z[i][j] = 0.0;
					}

				}
			}
			return z;
		}
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
 * 
 * @param a
 * @param axis
 * @param mean
 * @param sigma
 * @return
 */
	public static double[][] normalized(double[][] a, int axis, double[] mean,
			double[] sigma) {
		double[][] meanNormalized = null;
		meanNormalized = new double[a.length][a[0].length];

		if (axis == 1) {

			for (int i = 0; i < a.length; i++) {

				for (int j = 0; j < a[0].length; j++) {
					if (sigma[i] != 0) {
						meanNormalized[i][j] = (a[i][j] - mean[i]) / sigma[i];
					} else {
						meanNormalized[i][j] = (a[i][j] - mean[i]);
					}

				}

			}

		} else if (axis == 0) {

			for (int i = 0; i < a[0].length; ++i) {

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
 * 
 * @param a
 * @param axis
 * @return
 */
	public static Object[] normalizedWithMeanSigmaReturn(double[][] a, int axis) {
		double[][] meanNormalized = null;
		meanNormalized = new double[a.length][a[0].length];
		double[] mean = null;
		double[] sigma = null;
		Object[] result = new Object[3];
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
		result[0] = meanNormalized;
		result[1] = mean;
		result[2] = sigma;
		return result;

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
/**
 * 
 * @param z
 * @return
 */
	public static double max(double[][] z) {
		if (isGpuEnabled) {
			return MatrixUtilGPU.max(z);
		} else {
			double max = Double.MIN_VALUE;
			for (int i = 0; i < z.length; i++) {
				for (int j = 0; j < z[0].length; j++) {
					if (z[i][j] > max) {
						max = z[i][j];
					}
				}
			}

			return max;
		}
	}
/**
 * 
 * @param val
 */
	public static void print(String val) {
		System.out.println(val);
	}
/**
 * 
 * @param list
 */
	public static void printshapes(double[][]... list) {
		for (double[][] obj : list)
			System.out.println(obj + ": " + MatrixUtil.shape(obj));

	}
/**
 * 
 * @param a
 * @return
 */
	public static int getIndexOfMax(double[] a) {
		int indx = -1;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < a.length; i++) {
			if (a[i] > max) {
				indx = i;
				max = a[i];
			}

		}
		return indx;
	}
/**
 * 
 * @param a
 * @param b
 */
	public static void assertion(double[][] a, double[][] b) {
		// assert (dZ.shape == Z.shape);
		// np.printshapes(a,b);
		try {
			if (!(a.length == b.length) && (a[0].length == b[0].length)) {
				MatrixUtil.print("a: " + MatrixUtil.shape(a) + "b: " + MatrixUtil.shape(b));
			}

			assert ((a.length == b.length) && (a[0].length == b[0].length));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			MatrixUtil.print("a: " + MatrixUtil.shape(a) + "b: " + MatrixUtil.shape(b));
			throw e;
		}

	}
/**
 * 
 * @param Y
 * @return
 */
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

		oneHotIdentityMarix = MatrixUtil.createIdentity(labels.size(), labels.size());
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
/**
 * 
 * @param graphAdjac
 * @param noOfDimenstions
 * @return
 * @throws Exception
 */
	public static double[][] getGraphAdjacencyAfterNthRandomWalkAndDimension(
			float[][] graphAdjac, int noOfDimenstions) throws Exception {
		float[][] adjcAfterNthRandomWalk = null;
		double[][] adjcProbFeaureMatrix = null;
		float[][][] tmp = null;
		int numberOfCurrentDimensions = 0;
		int countNumberOfRandomWalks = 0;
		try {
			double sumofAllElements = 0;

			System.out.println("Adj matrix dimenstion: " + MatrixUtil.shape(graphAdjac)
					+ " numberof dimensions: " + noOfDimenstions);
			adjcAfterNthRandomWalk = graphAdjac;
			int expectedRandomWalkSize = (int) Math
					.ceil((double) noOfDimenstions / graphAdjac.length);
			System.out.println("Expected randomWalk " + expectedRandomWalkSize);
			tmp = new float[expectedRandomWalkSize][graphAdjac.length][graphAdjac.length];
			

			numberOfCurrentDimensions = graphAdjac.length;
			while (noOfDimenstions > numberOfCurrentDimensions) {
				adjcAfterNthRandomWalk = MatrixUtil.dot(adjcAfterNthRandomWalk,
						graphAdjac);
				countNumberOfRandomWalks++;
				numberOfCurrentDimensions = adjcAfterNthRandomWalk[0].length
						+ numberOfCurrentDimensions;
				sumofAllElements = MatrixUtil.sum(MatrixUtil.sum(adjcAfterNthRandomWalk, 0), 1)[0][0];
				
				tmp[countNumberOfRandomWalks] = MatrixUtil.divide(
						adjcAfterNthRandomWalk, (float) sumofAllElements);
				
			}
			int tmpLength = 0;
			adjcAfterNthRandomWalk = null;
			adjcProbFeaureMatrix = new double[graphAdjac.length][noOfDimenstions + 3]; // 3
																						// for
																						// extra
																						// temporal
																						// features;
			for (int i = 1; i < expectedRandomWalkSize; i++) {
				
				for (int matRow = 0; matRow < graphAdjac.length; matRow++) {
					for (int matCol = 0; matCol < graphAdjac[0].length; matCol++) {
						if ((tmpLength + matCol) <= noOfDimenstions) {
							adjcProbFeaureMatrix[matRow][tmpLength + matCol] = tmp[i][matRow][matCol];
						}

					}
				}
				
				tmpLength = tmpLength + tmp[i].length;
			}
			return adjcProbFeaureMatrix;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			tmp = null;
			adjcAfterNthRandomWalk = null;

		}

		return adjcProbFeaureMatrix;
	}
/**
 * 
 * @param graphAdjac
 * @param numberOFRandomWalk
 * @return
 * @throws Exception
 */
	public static double[][] getGraphAdjacencyAfterNthRandomWalk(
			float[][] graphAdjac, int numberOFRandomWalk) throws Exception {
		float[][] adjcAfterNthRandomWalk = null;
		double[][] adjcProbFeaureMatrix = null;
		float[][] tmp = null;
		int numberOfCurrentDimensions = 0;

		try {
			double sumofAllElements = 0;

			System.out.println("Adj matrix dimenstion: " + MatrixUtil.shape(graphAdjac)
					+ " number of random walks: " + numberOFRandomWalk);

			tmp = new float[graphAdjac.length][graphAdjac.length];
			
			numberOfCurrentDimensions = graphAdjac.length;
			
			adjcAfterNthRandomWalk = graphAdjac;
			for (int i = 0; i < numberOFRandomWalk; i++) {
				Stopwatch watch = new Stopwatch();
				adjcAfterNthRandomWalk = MatrixUtil.dot(adjcAfterNthRandomWalk,
						graphAdjac);
				System.out.println("Time taken in one dot: "
						+ watch.elapsedTime());
				numberOfCurrentDimensions = adjcAfterNthRandomWalk[0].length
						+ numberOfCurrentDimensions;
				watch = new Stopwatch();
				sumofAllElements = MatrixUtil.sum(MatrixUtil.sum(adjcAfterNthRandomWalk, 0), 1)[0][0];
				System.out.println("Time taken in sum operations: "
						+ watch.elapsedTime());
				tmp = MatrixUtil.divide(adjcAfterNthRandomWalk,
						(float) sumofAllElements);

			}
			int tmpLength = 0;
			adjcAfterNthRandomWalk = null;
			adjcProbFeaureMatrix = new double[graphAdjac.length][graphAdjac.length + 3]; // 3
																							// for
																							// extra
																							// temporal
																							// features;

			for (int matRow = 0; matRow < graphAdjac.length; matRow++) {
				for (int matCol = 0; matCol < graphAdjac[0].length; matCol++) {

					{
						adjcProbFeaureMatrix[matRow][matCol] = tmp[matRow][matCol];
					}

				}
			}

			return adjcProbFeaureMatrix;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			tmp = null;
			adjcAfterNthRandomWalk = null;

		}

		return adjcProbFeaureMatrix;
	}
/**
 * 
 * @param input
 * @return
 */
	public static double[] convertFloatsToDoubles(float[] input) {
		if (input == null) {
			return null; // Or throw an exception - your choice
		}
		double[] output = new double[input.length];
		for (int i = 0; i < input.length; i++) {
			output[i] = input[i];
		}
		return output;
	}
/**
 * 
 * @param input
 * @return
 */
	public static double[][] convertFloatsToDoubles(float[][] input) {
		if (input == null) {
			return null; // Or throw an exception - your choice
		}
		double[][] output = new double[input.length][input[0].length];
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				output[i][j] = input[i][j];
			}

		}
		return output;
	}
	
	
/**
 * Get determinant of a matrix
 * @param matrix
 * @return
 */
	
	private static double determinant(double[][] matrix) {
		if (matrix.length != matrix[0].length)
			throw new IllegalStateException("invalid dimensions");

		if (matrix.length == 2)
			return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

		double det = 0;
		for (int i = 0; i < matrix[0].length; i++)
			det += Math.pow(-1, i) * matrix[0][i]
					* determinant(minor(matrix, 0, i));
		return det;
	}
/**
 * Get determinant of a matrix
 * @param matrix
 * @return
 */
	private static float determinant(float[][] matrix) {
		if (matrix.length != matrix[0].length)
			throw new IllegalStateException("invalid dimensions");

		if (matrix.length == 2)
			return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

		float det = 0.0f;
		for (int i = 0; i < matrix[0].length; i++)
			det += Math.pow(-1, i) * matrix[0][i]
					* determinant(minor(matrix, 0, i));
		return det;
	}
/**
 * Get inverse of a matrix
 * @param matrix
 * @return
 */
	public static double[][] inverse(double[][] matrix) {
		double[][] inverse = new double[matrix.length][matrix.length];

		// minors and cofactors
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[i].length; j++)
				inverse[i][j] = Math.pow(-1, i + j)
						* determinant(minor(matrix, i, j));

		// adjugate and determinant
		double det = 1.0 / determinant(matrix);
		for (int i = 0; i < inverse.length; i++) {
			for (int j = 0; j <= i; j++) {
				double temp = inverse[i][j];
				inverse[i][j] = inverse[j][i] * det;
				inverse[j][i] = temp * det;
			}
		}

		return inverse;
	}
/**
 * Get inverse of a matrix
 * @param matrix
 * @return
 */
	public static float[][] inverse(float[][] matrix) {
		float[][] inverse = new float[matrix.length][matrix.length];

		// minors and cofactors
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[i].length; j++)
				inverse[i][j] = (float) Math.pow(-1, i + j)
						* determinant(minor(matrix, i, j));

		// adjugate and determinant
		float det = 1.0f / determinant(matrix);
		for (int i = 0; i < inverse.length; i++) {
			for (int j = 0; j <= i; j++) {
				float temp = inverse[i][j];
				inverse[i][j] = inverse[j][i] * det;
				inverse[j][i] = temp * det;
			}
		}

		return inverse;
	}
/**
 * Private
 * @param matrix
 * @param row
 * @param column
 * @return
 */
	private static double[][] minor(double[][] matrix, int row, int column) {
		double[][] minor = new double[matrix.length - 1][matrix.length - 1];

		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; i != row && j < matrix[i].length; j++)
				if (j != column)
					minor[i < row ? i : i - 1][j < column ? j : j - 1] = matrix[i][j];
		return minor;
	}
/**
 * Private
 * @param matrix
 * @param row
 * @param column
 * @return
 */
	private static float[][] minor(float[][] matrix, int row, int column) {
		float[][] minor = new float[matrix.length - 1][matrix.length - 1];

		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; i != row && j < matrix[i].length; j++)
				if (j != column)
					minor[i < row ? i : i - 1][j < column ? j : j - 1] = matrix[i][j];
		return minor;
	}
	/**
	 * Get sum of all elemts of a vector
	 * @param vector
	 * @return
	 * @throws InterruptedException
	 */
	public static double getSumOfAllElements(double[] vector)
			throws InterruptedException {
		
		final int length = vector.length;
		final int threads = length > IPredictionModelConstants.ARRAY_LENTH_PER_THREAD ? length
				/ IPredictionModelConstants.ARRAY_LENTH_PER_THREAD
				: 1;

		return (DoubleArraySumMultiThreaded.parallelSum(vector, threads));
	}
	
	
}