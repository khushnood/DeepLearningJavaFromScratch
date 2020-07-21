package com.jnn.utilities;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.jnn.consts.IPredictionModelConstants;
import com.jnn.enums.DeepNNActivations;
import com.jnn.enums.DeepNNCostFunctions;
import com.jnn.enums.DeepNNKeysForMaps;
import com.jnn.enums.DeepNNRegularizor;
import com.jnn.enums.ParameterInitializationType;
import com.jnn.enums.ProbablityDistributionTypes;

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
public class DNNUtils {
	/**
	 * 
	 * @param path
	 * @param encoding
	 * @return
	 */
	public static String open(String path, Charset encoding) {
		byte[] encoded = null;
		try {
			encoded = Files.readAllBytes(Paths.get(path));
		} catch (IOException ex) {
			Logger.getLogger(DNNUtils.class.getName()).log(Level.SEVERE, null,
					ex);
		}
		return new String(encoded, encoding);
	}

	/**
	 * 
	 * @param chars
	 * @return
	 */

	public static Map<Character, Integer> charToIx(char[] chars) {
		Map<Character, Integer> dict = new HashMap<>();
		for (int i = 0; i < chars.length; i++) {
			dict.put(chars[i], i);
		}
		return dict;
	}

	/**
	 * 
	 * @param chars
	 * @return
	 */
	public static Map<Integer, Character> ixToChar(char[] chars) {
		Map<Integer, Character> dict = new HashMap<>();
		for (int i = 0; i < chars.length; i++) {
			dict.put(i, chars[i]);
		}
		return dict;
	}

	/**
	 * 
	 * @param predictedLabel
	 * @param actualLabel
	 * @return
	 */

	public static double accuracyForClassification(double[][] predictedLabel,
			double[][] actualLabel) {
		int sum = 0;

		for (int i = 0; i < actualLabel[0].length; i++) {
			if ((predictedLabel[0][i] >= 0.5)
					&& (Double.compare(actualLabel[0][i], 1.0) == 0)) {
				sum++;
			} else if ((predictedLabel[0][i] < 0.5)
					&& (Double.compare(actualLabel[0][i], 0.0) == 0)) {
				sum++;
			}

		}
		return ((double) sum / actualLabel[0].length);
	}

	/**
	 * 
	 * @param predictedLabel
	 * @param actualLabel
	 * @return
	 */
	public static double accuracyForMultiClassClassification(
			double[] predictedLabel, double[] actualLabel) {
		int sum = 0;

		for (int i = 0; i < predictedLabel.length; i++) {
			if (Double.compare(actualLabel[i], predictedLabel[i]) == 0) {
				sum++;
			}

		}
		return ((double) sum / actualLabel.length);
	}

	/**
	 * 
	 * @param AL
	 * @param Y
	 * @return
	 */
	public static double accuracyForMultiClassClassification(double[][] AL,
			double[][] Y) {
		int sum = 0;

		double[][] AL_temp = MatrixUtil.T(AL);
		double[][] Y_temp = MatrixUtil.T(Y);
		MatrixUtil.assertion(AL, Y);
		for (int i = 0; i < Y_temp.length; i++) {

			int MaxElementIndex = MatrixUtil.getIndexOfMax(AL_temp[i]);
			if (MaxElementIndex == -1) {
				System.out.println(Arrays.toString(AL_temp[i]));
				continue;
			} else {
				// System.out.println(Arrays.toString(AL_temp[i]));

			}
			if (Double.compare(Y_temp[i][MaxElementIndex], 1.0) == 0) {
				sum++;
			}

		}
		return ((double) sum / Y[0].length);
	}

	/**
	 * 
	 * @param AL
	 * 
	 * @param labeltoOnhotVectorMap
	 * @return
	 */
	public static double[][] generateLabelsForMultiClassClassification(
			double[][] AL, Map<Double, double[]> labeltoOnhotVectorMap) {

		double[][] AL_temp = MatrixUtil.T(AL);

		double labels[][] = new double[AL_temp.length][2];

		for (int i = 0; i < AL_temp.length; i++) {

			int MaxElementIndex = MatrixUtil.getIndexOfMax(AL_temp[i]);
			if (MaxElementIndex == -1) {
				System.out.println(Arrays.toString(AL_temp[i]));
				continue;
			} else {

				labels[i][0] = i + 1;
				labels[i][1] = MatrixUtil.getLabelFromOneHotMatrix(
						labeltoOnhotVectorMap, MaxElementIndex);
			}

		}

		return (labels);
	}

	/**
	 * 
	 * @param predictedLabel
	 * @param actualLabel
	 * @param threshould
	 * @return
	 */

	public static double accuracyForClassification(double[][] predictedLabel,
			double[][] actualLabel, double threshould) {
		int sum = 0;

		for (int i = 0; i < actualLabel[0].length; i++) {
			if ((predictedLabel[0][i] >= threshould)
					&& (Double.compare(actualLabel[0][i], 1.0) == 0)) {
				sum++;
			} else if ((predictedLabel[0][i] < threshould)
					&& (Double.compare(actualLabel[0][i], 0.0) == 0)) {
				sum++;
			}

		}
		return ((double) sum / actualLabel[0].length);
	}

	/**
	 * 
	 * @param scanner
	 * @param matrix
	 * @param outPutLabel
	 * @param delimeter
	 * @param isdebug
	 */
	private static void readMatrix(final Scanner scanner,
			final double[][] matrix, final double[][] outPutLabel,
			String delimeter, boolean isdebug) {
		String line = null;

		String[] oneRow = null;
		String skippedLines = "";
		double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
		try {
			for (int i = 0; i < outPutLabel.length; ++i) {
				if (isdebug) {
					System.out.println("reading line :" + (i + 1));
				}

				if (scanner.hasNextLine()) {
					line = scanner.nextLine();
					oneRow = line.split(delimeter);
					for (int j = 0; j < matrix[i].length; ++j) {
						if (oneRow[j] != null && !oneRow[j].isEmpty()) {
							matrix[i][j] = Double.parseDouble(oneRow[j]);
						} else {
							skippedLines = skippedLines + (i + 1) + ", ";

							continue;
						}

					}
				}
				outPutLabel[i][0] = Double
						.parseDouble(oneRow[oneRow.length - 1]);
				if (Double.parseDouble(oneRow[oneRow.length - 1]) > max) {
					max = Double.parseDouble(oneRow[oneRow.length - 1]);
				} else if (Double.parseDouble(oneRow[oneRow.length - 1]) < min) {
					min = Double.parseDouble(oneRow[oneRow.length - 1]);
				}

			}
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if (isdebug) {
			System.out.println("some records skipped at lines " + skippedLines);
			System.out.println("minimum class label: " + min + " maxi: " + max);
		}
	}

	/**
	 * 
	 * @param scanner
	 * @param matrix
	 * @param delimeter
	 * @param isdebug
	 */
	private static void readMatrixForTestFeatureMatrixOnly(
			final Scanner scanner, final double[][] matrix, String delimeter,
			boolean isdebug) {
		String line = null;

		String[] oneRow = null;
		String skippedLines = "";
		double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
		try {
			for (int i = 0; i < matrix.length; ++i) {
				if (isdebug) {
					System.out.println("reading line :" + (i + 1));
				}

				if (scanner.hasNextLine()) {
					line = scanner.nextLine();
					oneRow = line.split(delimeter);
					for (int j = 0; j < matrix[i].length; ++j) {
						if (oneRow[j] != null && !oneRow[j].isEmpty()) {
							matrix[i][j] = Double.parseDouble(oneRow[j]);
						} else {
							skippedLines = skippedLines + (i + 1) + ", ";

							continue;
						}

					}
				}

				if (Double.parseDouble(oneRow[oneRow.length - 1]) > max) {
					max = Double.parseDouble(oneRow[oneRow.length - 1]);
				} else if (Double.parseDouble(oneRow[oneRow.length - 1]) < min) {
					min = Double.parseDouble(oneRow[oneRow.length - 1]);
				}

			}
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if (isdebug) {
			System.out.println("some records skipped at lines " + skippedLines);
			System.out.println("minimum class label: " + min + " maxi: " + max);
		}
	}

	/**
	 * 
	 * @param matrix
	 */
	private static void displayMatrix(final double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
	}

	/**
	 * 
	 * @param filePath
	 * @param delimeter
	 * @param isdebug
	 * @param trainRatio
	 * @param isClassification
	 * @return
	 * @throws FileNotFoundException
	 * @throws IllegalAccessException
	 */
	public static Map<Enum<DeepNNKeysForMaps>, Object> readFileAsMatrix(
			final String filePath, final String delimeter, boolean isdebug,
			double trainRatio, boolean isClassification)
			throws FileNotFoundException, IllegalAccessException {
		Scanner scanner = new Scanner(new File(filePath));

		int numbOfExamples = 1;
		String line = scanner.nextLine();
		String[] oneRow = null;
		Map<Enum<DeepNNKeysForMaps>, Object> resultMap = null;
		while (scanner.hasNextLine()) {
			scanner.nextLine();
			numbOfExamples++;
			oneRow = line.split(delimeter);
		}
		scanner.reset();
		scanner.close();
		scanner = null;

		final int n1 = numbOfExamples;
		final int n2 = oneRow.length - 1;
		scanner = new Scanner(new File(filePath));
		System.out.print(String.format("The matrix is %d x %d \n", n1, n2));

		double[][] X = new double[n1][n2];

		double[][] Y = new double[n1][1];
		if (isdebug) {
			System.out.println("Reading data to first matrix");
		}

		readMatrix(scanner, X, Y, delimeter, isdebug);
		if (isdebug) {
			System.out.println("Finished Matrix");
		}
		if (isClassification) {
			resultMap = MatrixUtil.splitIntoTrainAndTestForClassification(X, Y,
					trainRatio);
		} else {
			resultMap = MatrixUtil.splitIntoTrainAndTestForRegression(X, Y,
					trainRatio);
		}

		resultMap.put(DeepNNKeysForMaps.KEY_FOR_NUMBER_EXAMPLES, n1);
		resultMap.put(DeepNNKeysForMaps.KEY_FOR_NUMBEROFDIMENSIONS, n2);
		return (resultMap);
	}

	/**
	 * 
	 * @param filePath
	 * @param delimeter
	 * @param isdebug
	 * @return
	 * @throws FileNotFoundException
	 */
	public static double[][] readFileAsMatrixForTestFeaturesOnly(
			final String filePath, final String delimeter, boolean isdebug)
			throws FileNotFoundException {
		Scanner scanner = new Scanner(new File(filePath));

		int numbOfExamples = 1;
		String line = scanner.nextLine();
		String[] oneRow = null;
		Map<Enum<DeepNNKeysForMaps>, Object> resultMap = null;
		while (scanner.hasNextLine()) {
			scanner.nextLine();
			numbOfExamples++;
			oneRow = line.split(delimeter);
		}
		scanner.reset();
		scanner.close();
		scanner = null;

		final int n1 = numbOfExamples;
		final int n2 = oneRow.length;
		scanner = new Scanner(new File(filePath));
		System.out.print(String.format("The matrix is %d x %d \n", n1, n2));

		double[][] X_test = new double[n1][n2];

		if (isdebug) {
			System.out.println("Reading data to first matrix");
		}

		readMatrixForTestFeatureMatrixOnly(scanner, X_test, delimeter, isdebug);
		if (isdebug) {
			System.out.println("Finished Matrix");
		}

		return (X_test);
	}

	/**
	 * 
	 * @param layerDimensions
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> initializeParameterDeep(
			Map<Enum<DeepNNKeysForMaps>, Integer> layerDimensions)
			throws Exception {
		int numberOfDimenstions = (int) layerDimensions
				.get(DeepNNKeysForMaps.KEY_FOR_NUMBEROFDIMENSIONS);
		int nodes = (int) layerDimensions
				.get(DeepNNKeysForMaps.KEY_FOR_SIZE_OF_HIDDEN_LAYERS);
		int m = (int) layerDimensions
				.get(DeepNNKeysForMaps.KEY_FOR_NUMBER_EXAMPLES);
		MatrixUtil
				.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);
		Map<String, Object> parameters = new HashMap();
		int l = layerDimensions.size();
		for (int i = 0; i < l; i++) {
			parameters.put(String.valueOf("W_" + i),
					MatrixUtil.random(nodes, numberOfDimenstions));
			parameters.put(String.valueOf("b_" + i),
					MatrixUtil.random(nodes, m));
		}

		return parameters;

	}

	/**
	 * This method is used to initialized Neural network parameters using
	 * different methods. One can implement its own method.
	 * 
	 * @param layerDimensions
	 * @param numberOfTrainingExamples
	 * @param type
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> initializeParameterDeep(
			int[] layerDimensions, int numberOfTrainingExamples,
			ParameterInitializationType type) throws Exception {

		Map<String, Object> parameters = new HashMap();
		int LL = layerDimensions.length;
		double[][] tmp = null;
		if (type == ParameterInitializationType.BENGIO) {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);
				tmp = MatrixUtil.multiply(Math.sqrt(2), tmp);
				tmp = MatrixUtil
						.divide(tmp,
								(Math.sqrt(layerDimensions[i - 1]
										+ layerDimensions[i])));
				parameters.put(String.valueOf("W_" + i), tmp);

				parameters.put(String.valueOf("b_" + i),
						new double[layerDimensions[i]][1]);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		} else if (type == ParameterInitializationType.HE) {// He et al. 2015
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);
				tmp = MatrixUtil.multiply(Math.sqrt(2), tmp);
				tmp = MatrixUtil.divide(tmp,
						(Math.sqrt(layerDimensions[i - 1])));
				parameters.put(String.valueOf("W_" + i), tmp);

				parameters.put(String.valueOf("b_" + i),
						new double[layerDimensions[i]][1]);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;
				// assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

			}
		} else if (type == ParameterInitializationType.XAVIOUR) {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);
				// tmp = np.multiply(Math.sqrt(6), tmp);
				tmp = MatrixUtil.divide(tmp,
						(Math.sqrt(layerDimensions[i - 1])));
				parameters.put(String.valueOf("W_" + i), tmp);

				parameters.put(String.valueOf("b_" + i),
						new double[layerDimensions[i]][1]);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		} else if (type == ParameterInitializationType.OTHER) {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);
				tmp = MatrixUtil.multiply(Math.sqrt(6), tmp);
				tmp = MatrixUtil
						.divide(tmp,
								(Math.sqrt(layerDimensions[i - 1]
										+ layerDimensions[i])));
				parameters.put(String.valueOf("W_" + i), tmp);

				parameters.put(String.valueOf("b_" + i),
						new double[layerDimensions[i]][1]);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		} else if (type == ParameterInitializationType.NORMAL) {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1],
						ProbablityDistributionTypes.NORMAL);

				parameters.put(String.valueOf("W_" + i), tmp);

				parameters.put(String.valueOf("b_" + i),
						new double[layerDimensions[i]][1]);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		} else {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);

				parameters.put(String.valueOf("W_" + i), tmp);

				parameters.put(String.valueOf("b_" + i),
						new double[layerDimensions[i]][1]);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		}

		return parameters;

	}

	/**
	 * 
	 * @param layerDimensions
	 * @param numberOfTrainingExamples
	 * @param type
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> initializeParameterDeep_GCN(
			int[] layerDimensions, int numberOfTrainingExamples,
			ParameterInitializationType type) throws Exception {

		Map<String, Object> parameters = new HashMap();
		int LL = layerDimensions.length;
		double[][] tmp = null;
		if (type == ParameterInitializationType.BENGIO) {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);
				tmp = MatrixUtil.multiply(Math.sqrt(2), tmp);
				tmp = MatrixUtil
						.divide(tmp,
								(Math.sqrt(layerDimensions[i - 1]
										+ layerDimensions[i])));
				parameters.put(String.valueOf("W_" + i), tmp);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		} else if (type == ParameterInitializationType.HE) {// He et al. 2015
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);
				tmp = MatrixUtil.multiply(Math.sqrt(2), tmp);
				tmp = MatrixUtil.divide(tmp,
						(Math.sqrt(layerDimensions[i - 1])));
				parameters.put(String.valueOf("W_" + i), tmp);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		} else if (type == ParameterInitializationType.XAVIOUR) {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);

				tmp = MatrixUtil.divide(tmp,
						(Math.sqrt(layerDimensions[i - 1])));
				tmp = MatrixUtil.T(tmp);
				// np.shape(tmp);
				parameters.put(String.valueOf("W_" + i), tmp);

				// assert (tmp.length == layerDimensions[i] && tmp[0].length ==
				// layerDimensions[i - 1]);
				;

			}
		} else if (type == ParameterInitializationType.OTHER) {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);
				tmp = MatrixUtil.multiply(Math.sqrt(6), tmp);
				tmp = MatrixUtil
						.divide(tmp,
								(Math.sqrt(layerDimensions[i - 1]
										+ layerDimensions[i])));
				parameters.put(String.valueOf("W_" + i), tmp);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		} else if (type == ParameterInitializationType.NORMAL) {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1],
						ProbablityDistributionTypes.NORMAL);

				parameters.put(String.valueOf("W_" + i), tmp);

				parameters.put(String.valueOf("b_" + i),
						new double[layerDimensions[i]][1]);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		} else {
			for (int i = 1; i < (LL); i++) {
				tmp = MatrixUtil.random(layerDimensions[i],
						layerDimensions[i - 1]);

				parameters.put(String.valueOf("W_" + i), tmp);

				parameters.put(String.valueOf("b_" + i),
						new double[layerDimensions[i]][1]);

				assert (tmp.length == layerDimensions[i] && tmp[0].length == layerDimensions[i - 1]);
				;

			}
		}

		return parameters;

	}

	/**
	 * The adam optimizer uses its own parameters which needs to be initialized.
	 * 
	 * @param nnParameters
	 * @return
	 * @throws Exception
	 */
	public static Map<Enum<DeepNNKeysForMaps>, Object> initializeAdamParameters(
			Map<String, Object> nnParameters) throws Exception {

		Map<String, Object> adamParamDicV = new HashMap<>();
		Map<String, Object> adamParamDicS = new HashMap<>();
		Map<Enum<DeepNNKeysForMaps>, Object> finalMap = new HashMap<>();
		int l = nnParameters.size() / 2;
		double[][] tmpZeroslikeW = null;
		double[][] tmpZeroslikeb = null;
		for (int i = 0; i < (l); i++) {
			int mW = ((double[][]) nnParameters.get(String.valueOf("W_"
					+ (i + 1)))).length;
			int nW = ((double[][]) nnParameters.get(String.valueOf("W_"
					+ (i + 1))))[0].length;
			int mb = ((double[][]) nnParameters.get(String.valueOf("b_"
					+ (i + 1)))).length;
			int nb = ((double[][]) nnParameters.get(String.valueOf("b_"
					+ (i + 1))))[0].length;
			tmpZeroslikeW = MatrixUtil.zeros(mW, nW);
			tmpZeroslikeb = MatrixUtil.zeros(mb, nb);

			adamParamDicV.put(String.valueOf("dW_" + (i + 1)), tmpZeroslikeW);
			adamParamDicV.put(String.valueOf("db_" + (i + 1)), tmpZeroslikeb);

			adamParamDicS.put(String.valueOf("dW_" + (i + 1)), tmpZeroslikeW);
			adamParamDicS.put(String.valueOf("db_" + (i + 1)), tmpZeroslikeb);

		}
		finalMap.put(DeepNNKeysForMaps.KEY_FOR_ADAM_OPT_V, adamParamDicV);
		finalMap.put(DeepNNKeysForMaps.KEY_FOR_ADAM_OPT_S, adamParamDicS);

		return finalMap;

	}

	/**
	 * 
	 * @param A
	 * @param W
	 * @param b
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> linear_forward(double[][] A,
			double[][] W, double[][] b) throws Exception {
		double[][] Z = MatrixUtil.add(MatrixUtil.dot(W, A), b, true);
		Map<String, Object> cache = new HashMap();
		cache.put("A", A);
		cache.put("W", W);
		cache.put("b", b);
		cache.put("Z", Z);
		assert (Z.length == W.length && Z[0].length == A[0].length);
		return cache;

	}

	/**
	 * 
	 * @param Z
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> sigmoid(double[][] Z) throws Exception {

		double[][] A = MatrixUtil.sigmoid(Z);
		double[][] cache = Z;
		Map<String, Object> result = new HashMap();
		result.put("A", A);
		result.put("Z", cache);
		MatrixUtil.assertion(A, Z);
		return result;

	}

	/**
	 * https://arxiv.org/pdf/1710.05941v1.pdf
	 * 
	 * @param Z
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> swish(double[][] Z) throws Exception {

		double[][] A = MatrixUtil.swish(Z);
		double[][] cache = Z;
		Map<String, Object> result = new HashMap();
		result.put("A", A);
		result.put("Z", cache);
		MatrixUtil.assertion(A, Z);
		return result;

	}

	/**
	 * 
	 * @param Z
	 * @return
	 */

	public static Map<String, Object> ktest(double[][] Z) {

		double[][] A = MatrixUtil.ktest(Z);
		double[][] cache = Z;
		Map<String, Object> result = new HashMap();
		result.put("A", A);
		result.put("Z", cache);
		MatrixUtil.assertion(A, Z);
		return result;

	}

	/**
	 * 
	 * @param Z
	 * @return
	 */
	public static Map<String, Object> relu(double[][] Z) {

		double[][] A = MatrixUtil.replaceNegativeValuesOnOtherArrayBasis(Z, Z,
				0.0);
		// System.out.println("+Shape of Z"+np.shape(Z)+" A: "+np.shape(A));
		double[][] cache = Z;
		Map<String, Object> result = new HashMap();
		result.put("A", A);
		result.put("Z", cache);
		MatrixUtil.assertion(A, Z);
		return result;

	}

	/**
	 * 
	 * @param Z
	 * @return
	 */
	public static float[][] relu(float[][] Z) {

		float[][] result = MatrixUtil.replaceNegativeValuesOnOtherArrayBasis(Z,
				Z, 0.0f);

		return result;

	}

	/**
	 * 
	 * @param Z
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> tanh(double[][] Z) throws Exception {

		double[][] A = MatrixUtil.tanh(Z);
		double[][] cache = Z;
		Map<String, Object> result = new HashMap();
		result.put("A", A);
		result.put("Z", cache);
		return result;

	}

	/**
	 * 
	 * @param Z
	 * @return
	 */
	public static Map<String, Object> identity(double[][] Z) {

		double[][] A = Z;
		double[][] cache = Z;
		Map<String, Object> result = new HashMap();
		result.put("A", A);
		result.put("Z", cache);
		return result;

	}

	/**
	 * 
	 * @param Z
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> softmax(double[][] Z) throws Exception {
		double[][] A = MatrixUtil.softmax(Z, 0);
		double[][] cache = Z;
		Map<String, Object> result = new HashMap();
		result.put("A", A);
		result.put("Z", cache);
		MatrixUtil.assertion(A, Z);
		return result;
	}

	/**
	 * 
	 * @param A_prev
	 * @param W
	 * @param b
	 * @param activationType
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> linear_activation_forward(
			double[][] A_prev, double[][] W, double[][] b,
			DeepNNActivations activationType) throws Exception {
		// System.out.println("W :"+np.shape(W)+" A: "+np.shape(A_prev)+" b:
		// "+np.shape(b));
		Map<String, Object> result = new HashMap();
		Map<String, Object> Z_AND_LINEAR_CACHE = null;
		Map<String, Object> A_AND_ACTIVATION_CACHE = null;

		double[][] Z = null;

		if (activationType == DeepNNActivations.SIGMOID) {
			Z_AND_LINEAR_CACHE = linear_forward(A_prev, W, b);

			Z = (double[][]) Z_AND_LINEAR_CACHE.get("Z");

			A_AND_ACTIVATION_CACHE = sigmoid(Z);

		} else if (activationType == DeepNNActivations.RELU) {
			Z_AND_LINEAR_CACHE = linear_forward(A_prev, W, b);

			Z = (double[][]) Z_AND_LINEAR_CACHE.get("Z");
			A_AND_ACTIVATION_CACHE = DNNUtils.relu(Z);
		} else if (activationType == DeepNNActivations.TANH) {
			Z_AND_LINEAR_CACHE = linear_forward(A_prev, W, b);
			Z = (double[][]) Z_AND_LINEAR_CACHE.get("Z");
			A_AND_ACTIVATION_CACHE = DNNUtils.tanh(Z);
		} else if (activationType == DeepNNActivations.IDENTITY) {
			Z_AND_LINEAR_CACHE = linear_forward(A_prev, W, b);
			Z = (double[][]) Z_AND_LINEAR_CACHE.get("Z");
			A_AND_ACTIVATION_CACHE = DNNUtils.identity(Z);
		} else if (activationType == DeepNNActivations.SOFTMAX) {
			Z_AND_LINEAR_CACHE = linear_forward(A_prev, W, b);
			Z = (double[][]) Z_AND_LINEAR_CACHE.get("Z");
			A_AND_ACTIVATION_CACHE = DNNUtils.softmax(Z);
		} else if (activationType == DeepNNActivations.SWISH) {
			Z_AND_LINEAR_CACHE = linear_forward(A_prev, W, b);
			Z = (double[][]) Z_AND_LINEAR_CACHE.get("Z");
			A_AND_ACTIVATION_CACHE = DNNUtils.swish(Z);
		} else if (activationType == DeepNNActivations.KTEST) {
			Z_AND_LINEAR_CACHE = linear_forward(A_prev, W, b);
			Z = (double[][]) Z_AND_LINEAR_CACHE.get("Z");
			A_AND_ACTIVATION_CACHE = DNNUtils.ktest(Z);
		}

		result.put("A", (double[][]) A_AND_ACTIVATION_CACHE.get("A"));
		result.put("linCache", Z_AND_LINEAR_CACHE);
		result.put("actCache", A_AND_ACTIVATION_CACHE);
		return result;

	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @param hiddenLayersActivation
	 * @param finalLayerActivation
	 * @param regulizer
	 * @param dropOutKeepThreshould
	 * @param dropOutMap
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_forward_Generic(double[][] X,
			Map<String, Object> parameters,
			DeepNNActivations hiddenLayersActivation,
			DeepNNActivations finalLayerActivation,
			DeepNNRegularizor regulizer, double dropOutKeepThreshould,
			Map<String, Object> dropOutMap) throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null, dropOutRandomMatrix = null;
		int L = parameters.size() / 2;

		for (int i = 1; i < L; i++) {

			A_prev = A;
			// np.print("Shape of A at layer: "+i+" "+np.shape(A));
			feedForwardResult = linear_activation_forward(A_prev,
					(double[][]) parameters.get(String.valueOf("W_" + i)),
					(double[][]) parameters.get(String.valueOf("b_" + i)),
					hiddenLayersActivation);
			A = (double[][]) feedForwardResult.get("A");
			if (regulizer == DeepNNRegularizor.DROP_OUT
					&& dropOutKeepThreshould < 1.0
					&& dropOutKeepThreshould > 0.0) {
				dropOutRandomMatrix = MatrixUtil.random(A.length, A[0].length);
				dropOutRandomMatrix = MatrixUtil
						.createBooleanMatrixForSomeThreshold(
								dropOutRandomMatrix, dropOutKeepThreshould);
				A = MatrixUtil.multiply(A, dropOutRandomMatrix);
				A = MatrixUtil.divide(A, dropOutKeepThreshould);
				MatrixUtil.assertion(A, dropOutRandomMatrix);
				// np.printshapes(A,dropOutRandomMatrix);
				dropOutMap.put(String.valueOf("dropOutBoolMatrix_" + (i)),
						dropOutRandomMatrix);
			}

			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				finalLayerActivation);// / final
										// / layer
										// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));
		// np.print("Shape of A at layer: "+L+" : "+np.shape(A));
		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @param hiddenLayersActivation
	 * @param finalLayerActivation
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_forward_Generic(double[][] X,
			Map<String, Object> parameters,
			DeepNNActivations hiddenLayersActivation,
			DeepNNActivations finalLayerActivation) throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null, dropOutRandomMatrix = null;
		int L = parameters.size() / 2;

		for (int i = 1; i < L; i++) {

			A_prev = A;
			// np.print("Shape of A at layer: "+i+" "+np.shape(A));
			feedForwardResult = linear_activation_forward(A_prev,
					(double[][]) parameters.get(String.valueOf("W_" + i)),
					(double[][]) parameters.get(String.valueOf("b_" + i)),
					hiddenLayersActivation);
			A = (double[][]) feedForwardResult.get("A");

			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				finalLayerActivation);// / final
										// / layer
										// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));
		// np.print("Shape of A at layer: "+L+" : "+np.shape(A));
		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_forward_binary_classification(
			double[][] X, Map<String, Object> parameters) throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null;
		int L = parameters.size() / 2;

		for (int i = 1; i < L; i++) {

			A_prev = A;
			// np.print("Shape of A at layer: "+i+" "+np.shape(A));
			feedForwardResult = linear_activation_forward(A_prev,
					(double[][]) parameters.get(String.valueOf("W_" + i)),
					(double[][]) parameters.get(String.valueOf("b_" + i)),
					DeepNNActivations.TANH);
			A = (double[][]) feedForwardResult.get("A");
			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				DeepNNActivations.SIGMOID);// / final
											// / layer
											// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));
		// np.print("Shape of A at layer: "+L+" : "+np.shape(A));
		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_forward_multiClass_classification(
			double[][] X, Map<String, Object> parameters) throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null;
		int L = parameters.size() / 2;

		for (int i = 1; i < L; i++) {

			A_prev = A;
			feedForwardResult = linear_activation_forward(A_prev,
					(double[][]) parameters.get(String.valueOf("W_" + i)),
					(double[][]) parameters.get(String.valueOf("b_" + i)),
					DeepNNActivations.TANH);

			A = (double[][]) feedForwardResult.get("A");
			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				DeepNNActivations.SOFTMAX);// / final
											// / layer
											// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));

		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_forward_multiClass_classification_variedActivations(
			double[][] X, Map<String, Object> parameters) throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null;
		int L = parameters.size() / 2;

		for (int i = 1; i < L; i++) {

			A_prev = A;
			if (i % 2 == 0) {
				feedForwardResult = linear_activation_forward(A_prev,
						(double[][]) parameters.get(String.valueOf("W_" + i)),
						(double[][]) parameters.get(String.valueOf("b_" + i)),
						DeepNNActivations.IDENTITY);
			} else {
				feedForwardResult = linear_activation_forward(A_prev,
						(double[][]) parameters.get(String.valueOf("W_" + i)),
						(double[][]) parameters.get(String.valueOf("b_" + i)),
						DeepNNActivations.TANH);
			}

			A = (double[][]) feedForwardResult.get("A");
			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				DeepNNActivations.SOFTMAX);// / final
											// / layer
											// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));

		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @param dropOutKeepThreshould
	 * @param dropOutMap
	 * @return
	 * @throws Exception
	 */

	public static Map<String, Object> L_model_forward_multiClass_classification_withdropout(
			double[][] X, Map<String, Object> parameters,
			double dropOutKeepThreshould, Map<String, Object> dropOutMap)
			throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null;
		int L = parameters.size() / 2;
		double[][] dropOutRandomMatrix = null;
		for (int i = 1; i < L; i++) {

			A_prev = A;
			if (i % 2 == 0) {
				feedForwardResult = linear_activation_forward(A_prev,
						(double[][]) parameters.get(String.valueOf("W_" + i)),
						(double[][]) parameters.get(String.valueOf("b_" + i)),
						DeepNNActivations.TANH);
			} else {
				feedForwardResult = linear_activation_forward(A_prev,
						(double[][]) parameters.get(String.valueOf("W_" + i)),
						(double[][]) parameters.get(String.valueOf("b_" + i)),
						DeepNNActivations.TANH);
			}

			A = (double[][]) feedForwardResult.get("A");

			dropOutRandomMatrix = MatrixUtil.random(A.length, A[0].length);
			dropOutRandomMatrix = MatrixUtil
					.createBooleanMatrixForSomeThreshold(dropOutRandomMatrix,
							dropOutKeepThreshould);
			A = MatrixUtil.multiply(A, dropOutRandomMatrix);
			A = MatrixUtil.divide(A, dropOutKeepThreshould);
			MatrixUtil.assertion(A, dropOutRandomMatrix);
			// np.printshapes(A,dropOutRandomMatrix);
			dropOutMap.put(String.valueOf("dropOutBoolMatrix_" + (i)),
					dropOutRandomMatrix);

			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				DeepNNActivations.SOFTMAX);// / final
											// / layer
											// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));

		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @param dropOutKeepThreshould
	 * @param dropOutMap
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_forward_binary_classification_withDropOut(
			double[][] X, Map<String, Object> parameters,
			double dropOutKeepThreshould, Map<String, Object> dropOutMap)
			throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null;
		int L = parameters.size() / 2;
		double[][] dropOutRandomMatrix = null;

		for (int i = 1; i < L; i++) {

			A_prev = A;
			feedForwardResult = linear_activation_forward(A_prev,
					(double[][]) parameters.get(String.valueOf("W_" + i)),
					(double[][]) parameters.get(String.valueOf("b_" + i)),
					DeepNNActivations.RELU);
			A = (double[][]) feedForwardResult.get("A");
			// np.print("Shape of A at layer: " + i + " " + np.shape(A));
			// if(i>1)
			{
				dropOutRandomMatrix = MatrixUtil.random(A.length, A[0].length);
				dropOutRandomMatrix = MatrixUtil
						.createBooleanMatrixForSomeThreshold(
								dropOutRandomMatrix, dropOutKeepThreshould);
				A = MatrixUtil.multiply(A, dropOutRandomMatrix);
				A = MatrixUtil.divide(A, dropOutKeepThreshould);
				MatrixUtil.assertion(A, dropOutRandomMatrix);
				// np.printshapes(A,dropOutRandomMatrix);
				dropOutMap.put(String.valueOf("dropOutBoolMatrix_" + (i)),
						dropOutRandomMatrix);
			}

			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				DeepNNActivations.SIGMOID);// / final
											// / layer
											// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));
		double[][] tmp = (double[][]) ((Map<String, Object>) feedForwardResult
				.get(String.valueOf("actCache"))).get("A");
		// np.print("Shape of A at layer: " + L + " " + np.shape(tmp));
		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_forward_regression(double[][] X,
			Map<String, Object> parameters) throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null;
		int L = parameters.size() / 2;

		for (int i = 1; i < L; i++) {
			// System.out.println("at ithe layer: "+(i+1));
			A_prev = A;
			feedForwardResult = linear_activation_forward(A_prev,
					(double[][]) parameters.get(String.valueOf("W_" + i)),
					(double[][]) parameters.get(String.valueOf("b_" + i)),
					DeepNNActivations.IDENTITY);
			A = (double[][]) feedForwardResult.get("A");
			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				DeepNNActivations.IDENTITY);// / final
											// / layer
											// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));

		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_forward_ranking(double[][] X,
			Map<String, Object> parameters) throws Exception {

		Map<String, Object> resultCacheActivationMap = new HashMap();
		Map<String, Object> feedForwardResult = null;

		double[][] A = X;
		double[][] A_prev = null;
		int L = parameters.size() / 2;

		for (int i = 1; i < L; i++) {
			// System.out.println("at ithe layer: "+(i+1));
			A_prev = A;
			feedForwardResult = linear_activation_forward(A_prev,
					(double[][]) parameters.get(String.valueOf("W_" + i)),
					(double[][]) parameters.get(String.valueOf("b_" + i)),
					DeepNNActivations.RELU);
			A = (double[][]) feedForwardResult.get("A");
			resultCacheActivationMap.put(String.valueOf("linCache_" + i),
					feedForwardResult.get("linCache"));
			resultCacheActivationMap.put(String.valueOf("actCache_" + i),
					feedForwardResult.get("actCache"));

		}

		feedForwardResult = linear_activation_forward(A,
				(double[][]) parameters.get(String.valueOf("W_" + (L))),
				(double[][]) parameters.get(String.valueOf("b_" + (L))),
				DeepNNActivations.SOFTMAX);// / final
											// / layer
											// / activation
		resultCacheActivationMap.put(String.valueOf("linCache_" + (L)),
				feedForwardResult.get("linCache"));
		resultCacheActivationMap.put(String.valueOf("actCache_" + (L)),
				feedForwardResult.get("actCache"));

		return resultCacheActivationMap;
	}

	/**
	 * 
	 * @param dA
	 * @param cache
	 * @return
	 * @throws Exception
	 */
	public static double[][] sigmoid_backward(double[][] dA, double[][] cache)
			throws Exception {

		double[][] Z = cache;
		double[][] s = MatrixUtil.sigmoid(Z);

		// System.out.println("shape pf S: "+np.shape(s)+" dA: "+np.shape(dA)+"
		// cache:
		// "+np.shape(cache));
		double[][] dZ = MatrixUtil.multiply(MatrixUtil.multiply(dA, s),
				MatrixUtil.subtract(1, s)); // needs
		// recheck
		// if
		// dimensions
		// does'nt
		// match
		// np.printshapes(dA,cache,Z,s,dZ);
		MatrixUtil.assertion(Z, dZ);
		return dZ;

	}

	/**
	 * 
	 * @param dA
	 * @param cache
	 * @return
	 * @throws Exception
	 */
	public static double[][] swish_backward(double[][] dA, double[][] cache)
			throws Exception {

		double[][] Z = cache;
		double[][] s = MatrixUtil.swish(Z);
		double[][] sig = MatrixUtil.sigmoid(Z);

		double[][] dZtmp = MatrixUtil.add(s,
				MatrixUtil.multiply(sig, MatrixUtil.subtract(1, s)));
		double[][] dZ = MatrixUtil.multiply(dA, dZtmp);
		MatrixUtil.assertion(Z, dZ);
		return dZ;

	}

	/**
	 * This is implementation of our ktest activation during backpropogation.
	 * This code is completely modular, so here you just need to implement derivative of your own activation, in our case it was (x*tanh(x))
	 * 
	 * @param dA
	 * @param cache
	 * @return
	 * @throws Exception
	 */
	public static double[][] ktest_backward(double[][] dA, double[][] cache)
			throws Exception {

		double[][] Z = cache;
		double[][] ktest = MatrixUtil.ktest(Z);
		
		int n = Z.length;

		double[][] dZtmp1 = MatrixUtil.divide(MatrixUtil.multiply(2, ktest),
				MatrixUtil.exp(MatrixUtil.multiply(2, Z)));

		double[][] dZ = MatrixUtil.multiply(dA, dZtmp1);

		MatrixUtil.assertion(Z, dZ);
		return dZ;

	}

	/**
	 * 
	 * @param dA
	 * @param cache
	 * @return
	 * @throws Exception
	 */

	public static double[][] softMax_backward_old2(double[][] dA,
			double[][] cache) throws Exception {

		double[][] Z = cache;

		double[][] s = MatrixUtil.softmax(Z, 0);
		double[][] delta = MatrixUtil.createIdentity(s.length, s[0].length);
		double[][] dZ = MatrixUtil.multiply(MatrixUtil.multiply(dA, s),
				MatrixUtil.subtract(delta, s)); // needs
		
		MatrixUtil.assertion(Z, dZ);
		
		return dZ;

	}

	/**
	 * 
	 * @param dA
	 * @param cache
	 * @return
	 * @throws Exception
	 */
	public static double[][] softMax_backward(double[][] dA, double[][] cache)
			throws Exception {

		double[][] Z = cache;

		double[][] a = MatrixUtil.softmax(Z, 0);
		double[][] dA_a = MatrixUtil.sum(MatrixUtil.multiply((dA), a), 0);
	
		double[][] dZ = MatrixUtil.multiply(a,
				MatrixUtil.subtract(dA, dA_a, true)); // needs
	
		MatrixUtil.assertion(Z, dZ);
		return dZ;

	}

	/**
	 * 
	 * @param dA
	 * @param cache
	 * @return
	 * @throws Exception
	 */
	public static double[][] tanh_backward(double[][] dA, double[][] cache)
			throws Exception {

		double[][] Z = cache;
		double[][] s = MatrixUtil.tanh(Z);
	
		double[][] dZ = MatrixUtil.multiply(dA,
				MatrixUtil.subtract(1, MatrixUtil.multiply(s, s))); // needs
	
		MatrixUtil.assertion(Z, dZ);
		return dZ;

	}

	/**
	 * 
	 * @param dA
	 * @param cache
	 * @return
	 */

	public static double[][] identity_backward(double[][] dA, double[][] cache) {

		double[][] Z = cache;

		double[][] dZ = MatrixUtil.cloneArray(dA);
		
		MatrixUtil.assertion(Z, dZ);
		return dZ;

	}

	/**
	 * 
	 * @param dA
	 * @param cache
	 * @return
	 */

	public static double[][] relu_backward(double[][] dA, double[][] cache) {

		double[][] Z = cache;

		double[][] dZ = MatrixUtil.cloneArray(dA);

		dZ = MatrixUtil.replaceNegativeValuesOnOtherArrayBasis(dZ, Z, 0.0);
		
		MatrixUtil.assertion(Z, dZ);
		return dZ;

	}

	/**
	 * 
	 * @param dZ
	 * @param linCache
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> linear_backward(double[][] dZ,
			Map<String, Object> linCache) throws Exception {

		double[][] A_prev = (double[][]) linCache.get("A");
		double[][] W = (double[][]) linCache.get("W");
		double[][] b = (double[][]) linCache.get("b");
		int m = A_prev[0].length;
		double[][] dW = MatrixUtil.divide(
				MatrixUtil.dot(dZ, MatrixUtil.T(A_prev)), m);
		double[][] db = MatrixUtil.divide(MatrixUtil.sum(dZ, 1), m);
		double[][] dA_prev = MatrixUtil.dot(MatrixUtil.T(W), dZ);
		Map<String, Object> gradientsMap = new HashMap();
		// System.out.println("db: "+np.shape(db)+" b:"+np.shape(b));
		gradientsMap.put(String.valueOf("dA_prev"), dA_prev);
		gradientsMap.put(String.valueOf("dW"), dW);
		gradientsMap.put(String.valueOf("db"), db);

		MatrixUtil.assertion(dW, W);
		MatrixUtil.assertion(db, b);
		return gradientsMap;
	}

	/**
	 * Activation during backward operations. 
	 * @param dA
	 * @param linearCache
	 * @param activationCaches
	 * @param activationType
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> linear_activation_backward(double[][] dA,
			Map<String, Object> linearCache,
			Map<String, Object> activationCaches,
			DeepNNActivations activationType) throws Exception {
		double[][] dZ = null;
		Map<String, Object> resultGradientsMap = null;
		if (activationType == DeepNNActivations.RELU) {
			dZ = relu_backward(dA, (double[][]) activationCaches.get("Z"));
			resultGradientsMap = linear_backward(dZ, linearCache);
		} else if (activationType == DeepNNActivations.SIGMOID) {
			dZ = sigmoid_backward(dA, (double[][]) activationCaches.get("Z"));
			resultGradientsMap = linear_backward(dZ, linearCache);
		} else if (activationType == DeepNNActivations.TANH) {
			dZ = tanh_backward(dA, (double[][]) activationCaches.get("Z"));
			resultGradientsMap = linear_backward(dZ, linearCache);
		} else if (activationType == DeepNNActivations.IDENTITY) {
			dZ = identity_backward(dA, (double[][]) activationCaches.get("Z"));
			resultGradientsMap = linear_backward(dZ, linearCache);
		} else if (activationType == DeepNNActivations.SOFTMAX) {
			dZ = softMax_backward(dA, (double[][]) activationCaches.get("Z"));
			resultGradientsMap = linear_backward(dZ, linearCache);
		} else if (activationType == DeepNNActivations.SWISH) {
			dZ = swish_backward(dA, (double[][]) activationCaches.get("Z"));
			resultGradientsMap = linear_backward(dZ, linearCache);
		} else if (activationType == DeepNNActivations.KTEST) {
			dZ = ktest_backward(dA, (double[][]) activationCaches.get("Z"));
			resultGradientsMap = linear_backward(dZ, linearCache);
		}
		return resultGradientsMap;

	}

	/**
	 * 
	 * @param AL
	 * @param Y
	 * @param caches
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_backward_MultiClass_Classification(
			double[][] AL, double[][] Y, Map<String, Object> caches)
			throws Exception {
		Map<String, Object> grads = new HashMap<>();
		Map<String, Object> gradsMap = null;
		int L = caches.size() / 2;
		
		Y = MatrixUtil.reshape(Y, AL.length, AL[0].length);

		// initializing backprpogation

		double[][] dAL = MatrixUtil.subtract(AL, Y);
	
		Map<String, Object> currentLinCache = (Map<String, Object>) caches
				.get(String.valueOf("linCache_" + (L)));
		Map<String, Object> currentActCache = (Map<String, Object>) caches
				.get(String.valueOf("actCache_" + (L)));

		gradsMap = linear_activation_backward(dAL, currentLinCache,
				currentActCache, DeepNNActivations.SOFTMAX);

		grads.put(String.valueOf("dA_" + (L)), gradsMap.get("dA_prev"));
		grads.put(String.valueOf("dW_" + (L)), gradsMap.get("dW"));
		grads.put(String.valueOf("db_" + (L)), gradsMap.get("db"));

		Map<String, Object> gradsMapTmp = null;
		for (int nthLayer = (L - 1); nthLayer >= 1; nthLayer--) {
			currentLinCache = (Map<String, Object>) caches.get(String
					.valueOf("linCache_" + (nthLayer)));
			currentActCache = (Map<String, Object>) caches.get(String
					.valueOf("actCache_" + (nthLayer)));
			gradsMapTmp = linear_activation_backward(
					(double[][]) grads.get(String.valueOf("dA_"
							+ (nthLayer + 1))), currentLinCache,
					currentActCache, DeepNNActivations.TANH);

			grads.put(String.valueOf("dA_" + (nthLayer)),
					gradsMapTmp.get("dA_prev"));
			grads.put(String.valueOf("dW_" + (nthLayer)), gradsMapTmp.get("dW"));
			grads.put(String.valueOf("db_" + (nthLayer)), gradsMapTmp.get("db"));
		}

		return grads;
	}

	/**
	 * This method is for back propogation operations.
	 * 
	 * @param AL
	 * @param Y
	 * @param caches
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_backward_MultiClass_Classification_variedActivations(
			double[][] AL, double[][] Y, Map<String, Object> caches)
			throws Exception {
		Map<String, Object> grads = new HashMap<>();
		Map<String, Object> gradsMap = null;
		int L = caches.size() / 2;

		Y = MatrixUtil.reshape(Y, AL.length, AL[0].length);

		// initializing backprpogation

		double[][] dAL = MatrixUtil.subtract(AL, Y);

		Map<String, Object> currentLinCache = (Map<String, Object>) caches
				.get(String.valueOf("linCache_" + (L)));
		Map<String, Object> currentActCache = (Map<String, Object>) caches
				.get(String.valueOf("actCache_" + (L)));

		gradsMap = linear_activation_backward(dAL, currentLinCache,
				currentActCache, DeepNNActivations.SOFTMAX);

		grads.put(String.valueOf("dA_" + (L)), gradsMap.get("dA_prev"));
		grads.put(String.valueOf("dW_" + (L)), gradsMap.get("dW"));
		grads.put(String.valueOf("db_" + (L)), gradsMap.get("db"));

		Map<String, Object> gradsMapTmp = null;
		for (int nthLayer = (L - 1); nthLayer >= 1; nthLayer--) {
			currentLinCache = (Map<String, Object>) caches.get(String
					.valueOf("linCache_" + (nthLayer)));
			currentActCache = (Map<String, Object>) caches.get(String
					.valueOf("actCache_" + (nthLayer)));
			if (nthLayer % 2 == 0) {
				gradsMapTmp = linear_activation_backward(
						(double[][]) grads.get(String.valueOf("dA_"
								+ (nthLayer + 1))), currentLinCache,
						currentActCache, DeepNNActivations.IDENTITY);
			} else {
				gradsMapTmp = linear_activation_backward(
						(double[][]) grads.get(String.valueOf("dA_"
								+ (nthLayer + 1))), currentLinCache,
						currentActCache, DeepNNActivations.TANH);
			}

			grads.put(String.valueOf("dA_" + (nthLayer)),
					gradsMapTmp.get("dA_prev"));
			grads.put(String.valueOf("dW_" + (nthLayer)), gradsMapTmp.get("dW"));
			grads.put(String.valueOf("db_" + (nthLayer)), gradsMapTmp.get("db"));
		}

		return grads;
	}

	/**
	 * This method is for back propogation operations with drop outs.
	 * 
	 * @param AL
	 * @param Y
	 * @param caches
	 * @param dropOutKeepThreshould
	 * @param dropOutMap
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_backward_MultiClass_Classification_with_dropout(
			double[][] AL, double[][] Y, Map<String, Object> caches,
			double dropOutKeepThreshould, Map<String, Object> dropOutMap)
			throws Exception {
		Map<String, Object> grads = new HashMap<>();
		Map<String, Object> gradsMap = null;
		int L = caches.size() / 2;
		
		Y = MatrixUtil.reshape(Y, AL.length, AL[0].length);

		// initializing backprpogation
		double[][] tmpdA = null, dropOutBoolianMatrix = null;
		double[][] dAL = MatrixUtil.subtract(AL, Y);
		
		Map<String, Object> currentLinCache = (Map<String, Object>) caches
				.get(String.valueOf("linCache_" + (L)));
		Map<String, Object> currentActCache = (Map<String, Object>) caches
				.get(String.valueOf("actCache_" + (L)));

		gradsMap = linear_activation_backward(dAL, currentLinCache,
				currentActCache, DeepNNActivations.SOFTMAX);
		dropOutBoolianMatrix = (double[][]) dropOutMap.get("dropOutBoolMatrix_"
				+ (L - 1));
		tmpdA = (double[][]) gradsMap.get("dA_prev");
		tmpdA = MatrixUtil.multiply(tmpdA, dropOutBoolianMatrix);
		tmpdA = MatrixUtil.divide(tmpdA, dropOutKeepThreshould);

		grads.put(String.valueOf("dA_" + (L)), tmpdA);
		grads.put(String.valueOf("dW_" + (L)), gradsMap.get("dW"));
		grads.put(String.valueOf("db_" + (L)), gradsMap.get("db"));

		Map<String, Object> gradsMapTmp = null;
		for (int nthLayer = (L - 1); nthLayer >= 1; nthLayer--) {
			currentLinCache = (Map<String, Object>) caches.get(String
					.valueOf("linCache_" + (nthLayer)));
			currentActCache = (Map<String, Object>) caches.get(String
					.valueOf("actCache_" + (nthLayer)));

			if (nthLayer % 2 == 0) {
				gradsMapTmp = linear_activation_backward(
						(double[][]) grads.get(String.valueOf("dA_"
								+ (nthLayer + 1))), currentLinCache,
						currentActCache, DeepNNActivations.TANH);
			} else {
				gradsMapTmp = linear_activation_backward(
						(double[][]) grads.get(String.valueOf("dA_"
								+ (nthLayer + 1))), currentLinCache,
						currentActCache, DeepNNActivations.TANH);
			}

			tmpdA = (double[][]) gradsMapTmp.get("dA_prev");
			if (nthLayer > 1) {
				dropOutBoolianMatrix = (double[][]) dropOutMap
						.get("dropOutBoolMatrix_" + (nthLayer - 1));
				tmpdA = MatrixUtil.multiply(tmpdA, dropOutBoolianMatrix);
				tmpdA = MatrixUtil.divide(tmpdA, dropOutKeepThreshould);
			}

			grads.put(String.valueOf("dA_" + (nthLayer)), tmpdA);
			grads.put(String.valueOf("dW_" + (nthLayer)), gradsMapTmp.get("dW"));
			grads.put(String.valueOf("db_" + (nthLayer)), gradsMapTmp.get("db"));

		}

		return grads;
	}

	/**
	 * 
	 * @param AL
	 * @param Y
	 * @param caches
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_backward_BinaryClassification(
			double[][] AL, double[][] Y, Map<String, Object> caches)
			throws Exception {
		Map<String, Object> grads = new HashMap<>();

		int L = caches.size() / 2;

		Y = MatrixUtil.reshape(Y, AL.length, AL[0].length);

		// initializing backprpogation

		double[][] OneMinusY = MatrixUtil.subtract(1, Y);
		double[][] oneMinusAL = MatrixUtil.subtract(1, AL);
		double[][] divisionBoth = MatrixUtil.divide(OneMinusY, oneMinusAL);
		double[][] yAndAlDivision = MatrixUtil.divide(Y, AL);
		double[][] dAL = MatrixUtil.subtract(divisionBoth, yAndAlDivision);

		Map<String, Object> currentLinCache = (Map<String, Object>) caches
				.get(String.valueOf("linCache_" + (L)));
		Map<String, Object> currentActCache = (Map<String, Object>) caches
				.get(String.valueOf("actCache_" + (L)));

		Map<String, Object> gradsMap = linear_activation_backward(dAL,
				currentLinCache, currentActCache, DeepNNActivations.SIGMOID);

		grads.put(String.valueOf("dA_" + (L)), gradsMap.get("dA_prev"));
		grads.put(String.valueOf("dW_" + (L)), gradsMap.get("dW"));
		grads.put(String.valueOf("db_" + (L)), gradsMap.get("db"));

		Map<String, Object> gradsMapTmp = null;
		for (int nthLayer = (L - 1); nthLayer >= 1; nthLayer--) {
			currentLinCache = (Map<String, Object>) caches.get(String
					.valueOf("linCache_" + (nthLayer)));
			currentActCache = (Map<String, Object>) caches.get(String
					.valueOf("actCache_" + (nthLayer)));

			gradsMapTmp = linear_activation_backward(
					(double[][]) grads.get(String.valueOf("dA_"
							+ (nthLayer + 1))), currentLinCache,
					currentActCache, DeepNNActivations.TANH);
			grads.put(String.valueOf("dA_" + (nthLayer)),
					gradsMapTmp.get("dA_prev"));
			grads.put(String.valueOf("dW_" + (nthLayer)), gradsMapTmp.get("dW"));
			grads.put(String.valueOf("db_" + (nthLayer)), gradsMapTmp.get("db"));

		}

		return grads;
	}

	/**
	 * Get gradient of cost with respect to final layer activation.
	 * 
	 * @author Khushnood abbas
	 * @param AL
	 * @param Y
	 * @param costFunction
	 * @return
	 * @throws Exception
	 */
	public static double[][] getGradientOfCostAtFinalLayer(double[][] AL,
			double[][] Y, DeepNNCostFunctions costFunction) throws Exception {
		double[][] dAL = null;
		Y = MatrixUtil.reshape(Y, AL.length, AL[0].length);

		if (DeepNNCostFunctions.CROSS_ENTROPY == costFunction) {
			double[][] OneMinusY = MatrixUtil.subtract(1, Y);
			double[][] oneMinusAL = MatrixUtil.subtract(1, AL);
			double[][] divisionBoth = MatrixUtil.divide(OneMinusY, oneMinusAL);
			double[][] yAndAlDivision = MatrixUtil.divide(Y, AL);
			dAL = MatrixUtil.subtract(divisionBoth, yAndAlDivision);
		} else if (DeepNNCostFunctions.CATEGORICAL_CROSS_ENTROPY == costFunction) {

			dAL = MatrixUtil.multiply(-1, MatrixUtil.divide(Y, AL));
		} else if (DeepNNCostFunctions.MEAN_SQUARE == costFunction) {

			dAL = MatrixUtil.subtract(AL, Y);
		}

		else {
			throw new Exception("Declare cost gradient first for :"
					+ costFunction);
		}
		return dAL;
	}

	/**
	 * 
	 * @param AL
	 * @param Y
	 * @param caches
	 * @param hiddenLayersActivation
	 * @param finalLayerActivation
	 * @param costFunction
	 * @param regulizer
	 * @param dropOutKeepThreshould
	 * @param dropOutMap
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_backward_Generic(double[][] AL,
			double[][] Y, Map<String, Object> caches,
			DeepNNActivations hiddenLayersActivation,
			DeepNNActivations finalLayerActivation,
			DeepNNCostFunctions costFunction, DeepNNRegularizor regulizer,
			double dropOutKeepThreshould, Map<String, Object> dropOutMap)
			throws Exception {
		Map<String, Object> grads = new HashMap<>();
		double[][] dAL = null;
		int L = caches.size() / 2;

		double[][] tmpdA = null, dropOutBoolianMatrix = null;

		// initializing backprpogation

		dAL = DNNUtils.getGradientOfCostAtFinalLayer(AL, Y, costFunction);

		Map<String, Object> currentLinCache = (Map<String, Object>) caches
				.get(String.valueOf("linCache_" + (L)));
		Map<String, Object> currentActCache = (Map<String, Object>) caches
				.get(String.valueOf("actCache_" + (L)));

		Map<String, Object> gradsMap = linear_activation_backward(dAL,
				currentLinCache, currentActCache, finalLayerActivation);
		if (regulizer == DeepNNRegularizor.DROP_OUT
				&& dropOutKeepThreshould < 1.0 && dropOutKeepThreshould > 0.0) {

			dropOutBoolianMatrix = (double[][]) dropOutMap
					.get("dropOutBoolMatrix_" + (L - 1));
			tmpdA = (double[][]) gradsMap.get("dA_prev");

			tmpdA = MatrixUtil.multiply(tmpdA, dropOutBoolianMatrix);
			tmpdA = MatrixUtil.divide(tmpdA, dropOutKeepThreshould);

			grads.put(String.valueOf("dA_" + (L)), tmpdA);

		} else {
			grads.put(String.valueOf("dA_" + (L)), gradsMap.get("dA_prev"));
		}

		grads.put(String.valueOf("dW_" + (L)), gradsMap.get("dW"));
		grads.put(String.valueOf("db_" + (L)), gradsMap.get("db"));

		Map<String, Object> gradsMapTmp = null;
		for (int nthLayer = (L - 1); nthLayer >= 1; nthLayer--) {
			currentLinCache = (Map<String, Object>) caches.get(String
					.valueOf("linCache_" + (nthLayer)));
			currentActCache = (Map<String, Object>) caches.get(String
					.valueOf("actCache_" + (nthLayer)));

			gradsMapTmp = linear_activation_backward(
					(double[][]) grads.get(String.valueOf("dA_"
							+ (nthLayer + 1))), currentLinCache,
					currentActCache, hiddenLayersActivation);
			if (regulizer == DeepNNRegularizor.DROP_OUT
					&& dropOutKeepThreshould < 1.0
					&& dropOutKeepThreshould > 0.0) {
				tmpdA = (double[][]) gradsMapTmp.get("dA_prev");
				if (nthLayer > 1) {
					dropOutBoolianMatrix = (double[][]) dropOutMap
							.get("dropOutBoolMatrix_" + (nthLayer - 1));
					tmpdA = MatrixUtil.multiply(tmpdA, dropOutBoolianMatrix);
					tmpdA = MatrixUtil.divide(tmpdA, dropOutKeepThreshould);
				}

				grads.put(String.valueOf("dA_" + (nthLayer)), tmpdA);

			} else {
				grads.put(String.valueOf("dA_" + (nthLayer)),
						gradsMapTmp.get("dA_prev"));
			}

			grads.put(String.valueOf("dW_" + (nthLayer)), gradsMapTmp.get("dW"));
			grads.put(String.valueOf("db_" + (nthLayer)), gradsMapTmp.get("db"));

		}

		return grads;
	}

	/**
	 * 
	 * @param AL
	 * @param Y
	 * @param caches
	 * @param dropOutKeepThreshould
	 * @param dropOutMap
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_backward_withDropOut(
			double[][] AL, double[][] Y, Map<String, Object> caches,
			double dropOutKeepThreshould, Map<String, Object> dropOutMap)
			throws Exception {
		Map<String, Object> grads = new HashMap<>();

		int L = caches.size() / 2;

		Y = MatrixUtil.reshape(Y, AL.length, AL[0].length);

		double[][] OneMinusY = MatrixUtil.subtract(1, Y);
		double[][] oneMinusAL = MatrixUtil.subtract(1, AL);
		double[][] divisionBoth = MatrixUtil.divide(OneMinusY, oneMinusAL);
		double[][] yAndAlDivision = MatrixUtil.divide(Y, AL);
		double[][] dAL = MatrixUtil.subtract(divisionBoth, yAndAlDivision);
		double[][] dropOutBoolianMatrix = null;
		double[][] tmpdA = null;
		Map<String, Object> currentLinCache = (Map<String, Object>) caches
				.get(String.valueOf("linCache_" + (L)));
		Map<String, Object> currentActCache = (Map<String, Object>) caches
				.get(String.valueOf("actCache_" + (L)));

		Map<String, Object> gradsMap = linear_activation_backward(dAL,
				currentLinCache, currentActCache, DeepNNActivations.SIGMOID);
		dropOutBoolianMatrix = (double[][]) dropOutMap.get("dropOutBoolMatrix_"
				+ (L - 1));
		tmpdA = (double[][]) gradsMap.get("dA_prev");

		tmpdA = MatrixUtil.multiply(tmpdA, dropOutBoolianMatrix);
		tmpdA = MatrixUtil.divide(tmpdA, dropOutKeepThreshould);

		grads.put(String.valueOf("dA_" + (L)), tmpdA);
		grads.put(String.valueOf("dW_" + (L)), gradsMap.get("dW"));
		grads.put(String.valueOf("db_" + (L)), gradsMap.get("db"));

		Map<String, Object> gradsMapTmp = null;
		for (int nthLayer = (L - 1); nthLayer >= 1; nthLayer--) {
			currentLinCache = (Map<String, Object>) caches.get(String
					.valueOf("linCache_" + (nthLayer)));
			currentActCache = (Map<String, Object>) caches.get(String
					.valueOf("actCache_" + (nthLayer)));

		
			gradsMapTmp = linear_activation_backward(
					(double[][]) grads.get(String.valueOf("dA_"
							+ (nthLayer + 1))), currentLinCache,
					currentActCache, DeepNNActivations.RELU);
			tmpdA = (double[][]) gradsMapTmp.get("dA_prev");
			if (nthLayer > 1) {
				dropOutBoolianMatrix = (double[][]) dropOutMap
						.get("dropOutBoolMatrix_" + (nthLayer - 1));
			
				tmpdA = MatrixUtil.multiply(tmpdA, dropOutBoolianMatrix);
				tmpdA = MatrixUtil.divide(tmpdA, dropOutKeepThreshould);
			}

			grads.put(String.valueOf("dA_" + (nthLayer)), tmpdA);
			grads.put(String.valueOf("dW_" + (nthLayer)), gradsMapTmp.get("dW"));
			grads.put(String.valueOf("db_" + (nthLayer)), gradsMapTmp.get("db"));
		}

		return grads;
	}

	/**
	 * 
	 * @param AL
	 * @param Y
	 * @param caches
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_backward_ranking(double[][] AL,
			double[][] Y, Map<String, Object> caches) throws Exception {
		Map<String, Object> grads = new HashMap<>();
		Map<String, Object> gradsMap = null;
		int L = caches.size() / 2;

		Y = MatrixUtil.reshape(Y, AL.length, AL[0].length);

		// initializing backprpogation

		double[][] dAL = MatrixUtil.subtract(AL, Y);

		Map<String, Object> currentLinCache = (Map<String, Object>) caches
				.get(String.valueOf("linCache_" + (L)));
		Map<String, Object> currentActCache = (Map<String, Object>) caches
				.get(String.valueOf("actCache_" + (L)));

		gradsMap = linear_activation_backward(dAL, currentLinCache,
				currentActCache, DeepNNActivations.SOFTMAX);

		grads.put(String.valueOf("dA_" + (L)), gradsMap.get("dA_prev"));
		grads.put(String.valueOf("dW_" + (L)), gradsMap.get("dW"));
		grads.put(String.valueOf("db_" + (L)), gradsMap.get("db"));

		Map<String, Object> gradsMapTmp = null;
		for (int nthLayer = (L - 1); nthLayer >= 1; nthLayer--) {
			currentLinCache = (Map<String, Object>) caches.get(String
					.valueOf("linCache_" + (nthLayer)));
			currentActCache = (Map<String, Object>) caches.get(String
					.valueOf("actCache_" + (nthLayer)));

			gradsMapTmp = linear_activation_backward(
					(double[][]) grads.get(String.valueOf("dA_"
							+ (nthLayer + 1))), currentLinCache,
					currentActCache, DeepNNActivations.RELU);

			grads.put(String.valueOf("dA_" + (nthLayer)),
					gradsMapTmp.get("dA_prev"));
			grads.put(String.valueOf("dW_" + (nthLayer)), gradsMapTmp.get("dW"));
			grads.put(String.valueOf("db_" + (nthLayer)), gradsMapTmp.get("db"));
		}

		return grads;
	}

	/**
	 * 
	 * @param AL
	 * @param Y
	 * @param caches
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> L_model_backward_regression(
			double[][] AL, double[][] Y, Map<String, Object> caches)
			throws Exception {
		Map<String, Object> grads = new HashMap<>();

		int L = caches.size() / 2;

		Y = MatrixUtil.reshape(Y, AL.length, AL[0].length);

		// initializing backprpogation

		double[][] dAL = MatrixUtil.subtract(AL, Y);

		Map<String, Object> currentLinCache = (Map<String, Object>) caches
				.get(String.valueOf("linCache_" + (L)));
		Map<String, Object> currentActCache = (Map<String, Object>) caches
				.get(String.valueOf("actCache_" + (L)));

		Map<String, Object> gradsMap = linear_activation_backward(dAL,
				currentLinCache, currentActCache, DeepNNActivations.IDENTITY);

		grads.put(String.valueOf("dA_" + (L)), gradsMap.get("dA_prev"));
		grads.put(String.valueOf("dW_" + (L)), gradsMap.get("dW"));
		grads.put(String.valueOf("db_" + (L)), gradsMap.get("db"));

		Map<String, Object> gradsMapTmp = null;
		for (int nthLayer = (L - 1); nthLayer >= 1; nthLayer--) {
			currentLinCache = (Map<String, Object>) caches.get(String
					.valueOf("linCache_" + (nthLayer)));
			currentActCache = (Map<String, Object>) caches.get(String
					.valueOf("actCache_" + (nthLayer)));

			gradsMapTmp = linear_activation_backward(
					(double[][]) grads.get(String.valueOf("dA_"
							+ (nthLayer + 1))), currentLinCache,
					currentActCache, DeepNNActivations.IDENTITY);

			grads.put(String.valueOf("dA_" + (nthLayer)),
					gradsMapTmp.get("dA_prev"));
			grads.put(String.valueOf("dW_" + (nthLayer)), gradsMapTmp.get("dW"));
			grads.put(String.valueOf("db_" + (nthLayer)), gradsMapTmp.get("db"));
		}

		return grads;
	}

	/**
	 * updateParametersWithAdams
	 * @param parameters
	 * @param grads
	 * @param adamParameters
	 * @param t
	 * @param learning_rate
	 * @param beta1
	 * @param beta2
	 * @param epsilon
	 * @return
	 * @throws Exception
	 */
	public static Map<Enum<DeepNNKeysForMaps>, Object> updateParametersWithAdams(
			Map<String, Object> parameters, Map<String, Object> grads,
			Map<Enum<DeepNNKeysForMaps>, Object> adamParameters, double t,
			double learning_rate, double beta1, double beta2, double epsilon)
			throws Exception {
		int L = parameters.size() / 2;
		Map<String, Object> resultMap = new HashMap<>();
		Map<String, Object> V = null;
		Map<String, Object> S = null;
		Map<String, Object> V_Corrected = null;
		Map<String, Object> S_Corrected = null;
		V = (Map<String, Object>) adamParameters
				.get(DeepNNKeysForMaps.KEY_FOR_ADAM_OPT_V);
		S = (Map<String, Object>) adamParameters
				.get(DeepNNKeysForMaps.KEY_FOR_ADAM_OPT_S);

		for (int i = 0; i < L; i++) {
			double[][] V_dWtmp = (double[][]) V.get(String.valueOf("dW_"
					+ (i + 1)));
			double[][] S_dWtmp = (double[][]) S.get(String.valueOf("dW_"
					+ (i + 1)));
			double[][] V_dbtmp = (double[][]) V.get(String.valueOf("db_"
					+ (i + 1)));
			double[][] S_dbtmp = (double[][]) S.get(String.valueOf("db_"
					+ (i + 1)));
			double[][] dW_tmp = (double[][]) grads.get(String.valueOf("dW_"
					+ (i + 1)));
			double[][] db_tmp = (double[][]) grads.get(String.valueOf("db_"
					+ (i + 1)));
			V_dWtmp = MatrixUtil.add(MatrixUtil.multiply(beta1, V_dWtmp),
					MatrixUtil.multiply((1 - beta1), dW_tmp));
			V_dbtmp = MatrixUtil.add(MatrixUtil.multiply(beta1, V_dbtmp),
					MatrixUtil.multiply((1 - beta1), db_tmp));

			double[][] V_Corrected_dw = MatrixUtil.divide(V_dWtmp,
					(1 - Math.pow(beta1, t)));
			double[][] V_Corrected_db = MatrixUtil.divide(V_dbtmp,
					(1 - Math.pow(beta1, t)));

			S_dWtmp = MatrixUtil.add(
					MatrixUtil.multiply(beta2, S_dWtmp),
					MatrixUtil.multiply((1 - beta2),
							MatrixUtil.power(dW_tmp, 2)));
			S_dbtmp = MatrixUtil.add(
					MatrixUtil.multiply(beta2, S_dbtmp),
					MatrixUtil.multiply((1 - beta2),
							MatrixUtil.power(db_tmp, 2)));
			double[][] S_Corrected_dw = MatrixUtil.divide(S_dWtmp,
					(1 - Math.pow(beta2, t)));
			double[][] S_Corrected_db = MatrixUtil.divide(S_dbtmp,
					(1 - Math.pow(beta2, t)));

			double[][] W_tmp = (double[][]) parameters.get(String.valueOf("W_"
					+ (i + 1)));
			double[][] b_tmp = (double[][]) parameters.get(String.valueOf("b_"
					+ (i + 1)));

			W_tmp = MatrixUtil.subtract(W_tmp, MatrixUtil.multiply(
					learning_rate, MatrixUtil.divide(V_Corrected_dw,
							(MatrixUtil.sqrt(MatrixUtil.add(S_Corrected_dw,
									epsilon))))));
			b_tmp = MatrixUtil.subtract(b_tmp, MatrixUtil.multiply(
					learning_rate, MatrixUtil.divide(V_Corrected_db,
							(MatrixUtil.sqrt(MatrixUtil.add(S_Corrected_db,
									epsilon))))));

			resultMap.put(String.valueOf("W_" + (i + 1)), W_tmp);
			resultMap.put(String.valueOf("b_" + (i + 1)), b_tmp);
			V.put(String.valueOf("dW_" + (i + 1)), V_dWtmp);
			V.put(String.valueOf("db_" + (i + 1)), V_dbtmp);
			S.put(String.valueOf("dW_" + (i + 1)), S_dWtmp);
			S.put(String.valueOf("db_" + (i + 1)), S_dbtmp);

		}
		adamParameters.put(DeepNNKeysForMaps.KEY_FOR_ADAM_OPT_V, V);
		adamParameters.put(DeepNNKeysForMaps.KEY_FOR_ADAM_OPT_S, S);
		adamParameters.put(DeepNNKeysForMaps.KEY_FOR_NN_PARAMETERS, resultMap);

		return adamParameters;

	}

	/**
	 * updateParametersWithGD
	 * @param parameters
	 * @param grads
	 * @param learning_rate
	 * @return
	 * @throws Exception
	 */
	public static Map<String, Object> updateParametersWithGD(
			Map<String, Object> parameters, Map<String, Object> grads,
			double learning_rate) throws Exception {

		int L = parameters.size() / 2;
		Map<String, Object> resultMap = new HashMap<>();
		for (int i = 0; i < L; i++) {
			double[][] W_tmp = (double[][]) parameters.get(String.valueOf("W_"
					+ (i + 1)));
			double[][] b_tmp = (double[][]) parameters.get(String.valueOf("b_"
					+ (i + 1)));
			double[][] dW_tmp = (double[][]) grads.get(String.valueOf("dW_"
					+ (i + 1)));
			double[][] db_tmp = (double[][]) grads.get(String.valueOf("db_"
					+ (i + 1)));
			W_tmp = MatrixUtil.subtract(W_tmp,
					MatrixUtil.multiply(learning_rate, dW_tmp));
			b_tmp = MatrixUtil.subtract(b_tmp,
					MatrixUtil.multiply(learning_rate, db_tmp));
			resultMap.put(String.valueOf("W_" + (i + 1)), W_tmp);
			resultMap.put(String.valueOf("b_" + (i + 1)), b_tmp);
		}

		return resultMap;

	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @param hiddenLayersActivation
	 * @param finalLayerActivation
	 * @return
	 * @throws Exception
	 */
	public static double[][] getPredictedActivationScoreForGenericActivations(
			double[][] X, Map<String, Object> parameters,
			DeepNNActivations hiddenLayersActivation,
			DeepNNActivations finalLayerActivation) throws Exception {
		double[][] AL = null;
		Map<String, Object> feedForwardResult = null;
		Map<String, Object> actCaches = null;
		double[][] A = X;

		int L = parameters.size() / 2;
		feedForwardResult = DNNUtils.L_model_forward_Generic(X, parameters,
				hiddenLayersActivation, finalLayerActivation);
		actCaches = (Map<String, Object>) feedForwardResult.get(String
				.valueOf("actCache_" + (L)));
		AL = (double[][]) actCaches.get(String.valueOf("A"));

		return AL;
	}

	/**
	 * 
	 * @param A_hat
	 * @param D_hat
	 * @param X
	 * @param numberOfDimenstions
	 * @return
	 * @throws Exception
	 */
	public static double[][] getGraphRepresentatinoAfterGraphConvolutionOperations(
			float[][] A_hat, float[][] D_hat, float[][] X,
			int numberOfDimenstions) throws Exception {

		float[][] W_1 = MatrixUtil.randomF(A_hat.length,
				numberOfDimenstions * 2);

		float[][] W_2 = MatrixUtil.randomF(W_1[0].length, numberOfDimenstions);

		float[][] H_1 = getGCNLayer(A_hat, D_hat, X, W_1);

		float[][] H_2 = getGCNLayer(A_hat, D_hat, H_1, W_2);

		return MatrixUtil.convertFloatsToDoubles(H_2);
	}

	/**
	 * 
	 * @param A_hat
	 * @param D_hat
	 * @param X
	 * @param W
	 * @return
	 * @throws Exception
	 */
	public static float[][] getGCNLayer(float[][] A_hat, float[][] D_hat,
			float[][] X, float[][] W) throws Exception {
		// Tmp=D_hat**-1 * A_hat * X * W
		float[][] tmp = MatrixUtil.inverse(D_hat);
		tmp = MatrixUtil.dot(tmp, A_hat);
		tmp = MatrixUtil.dot(tmp, X);
		tmp = MatrixUtil.dot(tmp, W);
		return relu(tmp);

	}

	/**
	 * 
	 * @param A_hat
	 * @param X
	 * @param W
	 * @return
	 * @throws Exception
	 */
	public static double[][] getGCNForward(double[][] A_hat, double[][] X,
			double[][] W) throws Exception {
		// Tmp=D_hat**-1 * A_hat * X * W
		

		return MatrixUtil.tanh(MatrixUtil.dot(MatrixUtil.dot(A_hat, X), W));

	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public static double[][] getPredictedResult_multiclass_classification(
			double[][] X, Map<String, Object> parameters) throws Exception {
		double[][] AL = null;
		Map<String, Object> feedForwardResult = null;
		Map<String, Object> actCaches = null;
		double[][] A = X;

		int L = parameters.size() / 2;
		feedForwardResult = DNNUtils.L_model_forward_multiClass_classification(
				X, parameters);
		actCaches = (Map<String, Object>) feedForwardResult.get(String
				.valueOf("actCache_" + (L)));
		AL = (double[][]) actCaches.get(String.valueOf("A"));

		return AL;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public static double[][] getPredictedResult_multiclass_classification_variedActivations(
			double[][] X, Map<String, Object> parameters) throws Exception {
		double[][] AL = null;
		Map<String, Object> feedForwardResult = null;
		Map<String, Object> actCaches = null;
		double[][] A = X;

		int L = parameters.size() / 2;
		feedForwardResult = DNNUtils
				.L_model_forward_multiClass_classification_variedActivations(X,
						parameters);
		actCaches = (Map<String, Object>) feedForwardResult.get(String
				.valueOf("actCache_" + (L)));
		AL = (double[][]) actCaches.get(String.valueOf("A"));

		return AL;
	}

	/**
	 * 
	 * @param X
	 * @param learnedParametersForMultiClass
	 * @param hiddenLayersActivation
	 * @param finalLayerActivation
	 * @return
	 * @throws Exception
	 */
	public static double[] getPredictedResult_multi_class_classification_one_vs_all(
			double[][] X,
			Map<Double, Map<String, Object>> learnedParametersForMultiClass,
			DeepNNActivations hiddenLayersActivation,
			DeepNNActivations finalLayerActivation) throws Exception {
		double[][] AL = null;

		Map<String, Object> feedForwardResult = null;
		Map<String, Object> actCaches = null;

		Map<String, Object> parameters = null;
		int numberOfExamples = X[0].length;
		double[] predictedLabel = new double[numberOfExamples];
		double[][] allALs = new double[learnedParametersForMultiClass.keySet()
				.size()][numberOfExamples];
		int labelCount = 1;
		Map<Integer, Double> matIndexLabelCountMap = new HashMap<>();
		for (double uniqueLabel : learnedParametersForMultiClass.keySet()) {
			parameters = learnedParametersForMultiClass.get(uniqueLabel);

			int L = parameters.size() / 2;
			feedForwardResult = DNNUtils.L_model_forward_Generic(X, parameters,
					hiddenLayersActivation, finalLayerActivation);
			actCaches = (Map<String, Object>) feedForwardResult.get(String
					.valueOf("actCache_" + (L)));
			AL = (double[][]) actCaches.get(String.valueOf("A"));
			allALs[labelCount - 1] = AL[0];
			matIndexLabelCountMap.put(labelCount, uniqueLabel);
			labelCount++;
		}

		for (int numEx = 0; numEx < numberOfExamples; numEx++) {
			double max = Double.MIN_VALUE;
			for (int labelCountIndx : matIndexLabelCountMap.keySet()) {
				if (max < allALs[labelCountIndx - 1][numEx]) {
					max = allALs[labelCountIndx - 1][numEx];
					predictedLabel[numEx] = matIndexLabelCountMap
							.get(labelCountIndx);
				}
			}
		}

		return predictedLabel;
	}

	/**
	 * 
	 * @param X
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public static double[][] getPredictedResult_regression(double[][] X,
			Map<String, Object> parameters) throws Exception {
		double[][] AL = null;
		Map<String, Object> feedForwardResult = null;
		Map<String, Object> actCaches = null;
		double[][] A = X;

		int L = parameters.size() / 2;
		feedForwardResult = DNNUtils.L_model_forward_regression(X, parameters);
		actCaches = (Map<String, Object>) feedForwardResult.get(String
				.valueOf("actCache_" + (L)));
		AL = (double[][]) actCaches.get(String.valueOf("A"));

		return AL;
	}

	/**
	 * 
	 * @param predictedValues
	 * @param groundTruthValues
	 * @return
	 * @throws Exception
	 */
	public static double getMSE(double[] predictedValues,
			double[] groundTruthValues) throws Exception {
		double sum = 0;
		int validCound = 0;

		for (int countListB = 0; countListB < groundTruthValues.length; countListB++) {
			if (Double.compare(groundTruthValues[countListB], 0.0) == 1) {

				validCound++;
				sum = sum
						+ ((predictedValues[countListB] - groundTruthValues[countListB]) * (predictedValues[countListB] - groundTruthValues[countListB]));

			}
		}

		return !Double.isNaN(sum) ? (sum / validCound) : Double.MAX_VALUE;
	}

	/**
	 * 
	 * @param batch_size
	 * @param Y
	 * @param A
	 * @param costFunction
	 * @return
	 * @throws Exception
	 */
	public static double getCost(int batch_size, double[][] Y, double[][] A,
			DeepNNCostFunctions costFunction) throws Exception {
		double cost = Double.MAX_VALUE;
		if (DeepNNCostFunctions.CROSS_ENTROPY == costFunction) {
			cost = DNNUtils.cross_entropy(batch_size, Y, A);
		} else if (DeepNNCostFunctions.CATEGORICAL_CROSS_ENTROPY == costFunction) {
			cost = DNNUtils.sparse_categorial_crossentropy(batch_size, Y, A);
		} else {
			new Exception("Method not implemented");
		}
		return cost;
	}

	/**
	 * 
	 * @param batch_size
	 * @param Y
	 * @param A
	 * @return
	 * @throws Exception
	 */
	public static double cross_entropy(int batch_size, double[][] Y,
			double[][] A) throws Exception {
		int m = A.length;
		int n = A[0].length;

		MatrixUtil.assertion(Y, A);
		double[][] z = null;

		double[][] tmpA = MatrixUtil.dot(Y, MatrixUtil.T(MatrixUtil.log(A)));
		double[][] tmpB = MatrixUtil.dot(MatrixUtil.subtract(1, Y),
				MatrixUtil.T(MatrixUtil.log(MatrixUtil.subtract(1, A))));
		z = MatrixUtil.add(tmpA, tmpB);

		return -z[0][0] / batch_size;

	}

	/**
	 * 
	 * @param batch_size
	 * @param Y
	 * @param A
	 * @return
	 */
	public static double sparse_categorial_crossentropy(int batch_size,
			double[][] Y, double[][] A) {

		MatrixUtil.assertion(Y, A);
		
		int m = A.length;
		int n = A[0].length;
		double[][] z = new double[m][n];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				z[j][i] = (Y[j][i] * Math.log(A[j][i]));
				
			}
		}

		double sum = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				sum += z[j][i];
			}
		}
		return -sum / batch_size;
	}

	/**
	 * 
	 * @param batch_size
	 * @param Y
	 * @param A
	 * @return
	 */
	public static double cross_entropy_withRegularization(int batch_size,
			double[][] Y, double[][] A) {
		int m = A.length;
		int n = A[0].length;

		double[][] z = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				z[i][j] = (Y[i][j] * Math.log(A[i][j]))
						+ ((1 - Y[i][j]) * Math.log(1 - A[i][j]));
			}
		}

		double sum = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum += z[i][j];
			}
		}
		return -sum / batch_size;
	}

	/**
	 * 
	 * @param batch_size
	 * @param Y
	 * @param A
	 * @return
	 */
	public static double cost_mse(int batch_size, double[][] Y, double[][] A) {
		int m = A.length;
		int n = A[0].length;

		double[][] z = new double[m][1];
		for (int i = 0; i < m; i++) {
			// for (int j = 0; j < n; j++) {
			z[i][0] = (A[i][0] - Y[i][0]) * (A[i][0] - Y[i][0]);
			// }
		}

		double sum = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < 1; j++) {
				sum += z[i][j];
			}
		}
		return sum / (2 * batch_size);
	}
}
