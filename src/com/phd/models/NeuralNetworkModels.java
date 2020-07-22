package com.phd.models;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.jnn.enums.DeepNNActivations;
import com.jnn.enums.DeepNNCostFunctions;
import com.jnn.enums.DeepNNKeysForMaps;
import com.jnn.enums.DeepNNOptimizers;
import com.jnn.enums.DeepNNRegularizor;
import com.jnn.enums.ParameterInitializationType;
import com.jnn.utilities.DNNUtils;
import com.jnn.utilities.MatrixUtil;
import com.jnn.utilities.Stopwatch;
//import org.jfree.data.xy.XYSeries; This was for plotting cost
//import org.jfree.data.xy.XYSeriesCollection;
import com.jnn.consts.IPredictionModelConstants;


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
 */

public class NeuralNetworkModels {
/**
 * This two layer model is for learning how neuaral network works. This is from Coursera andrew Ng course. 
 Initial some idea I got from this author https://jeraldy.git for two layer model.
 * @param X
 * @param Y
 * @param layerDimensions
 * @param learningRate
 * @param numOfIterations
 * @param iprintCost
 * @return
 * @throws Exception
 */
	public static Map<String, Object> twoLayerModel(double[][] X, double[][] Y,
			int[] layerDimensions, double learningRate, int numOfIterations,
			boolean iprintCost) throws Exception {
		Map<String, Object> learnedParameters = null;
		X = MatrixUtil.T(X);
		Y = MatrixUtil.T(Y);
		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);
		int m = X[0].length;

		int n_y = layerDimensions[2];

		double[][] W1 = MatrixUtil.random(layerDimensions[1], layerDimensions[0]);
		double[][] b1 = new double[layerDimensions[1]][m];

		double[][] W2 = MatrixUtil.random(n_y, layerDimensions[1]);
		double[][] b2 = new double[n_y][m];

		for (int i = 0; i < numOfIterations; i++) {
			// Foward Prop
			// LAYER 1
			double[][] Z1 = MatrixUtil.add(MatrixUtil.dot(W1, X), b1);
			double[][] A1 = MatrixUtil.sigmoid(Z1);
			// double[][] A1 = Np.relu(Z1);
			// LAYER 2
			double[][] Z2 = MatrixUtil.add(MatrixUtil.dot(W2, A1), b2);
			double[][] A2 = MatrixUtil.sigmoid(Z2);

			double cost = DNNUtils.cross_entropy(m, Y, A2);
			// costs.getData().add(new XYChart.Data(i, cost));
			// Back Prop
			// LAYER 2
			double[][] dZ2 = MatrixUtil.subtract(A2, Y);
			double[][] dW2 = MatrixUtil.divide(MatrixUtil.dot(dZ2, MatrixUtil.T(A1)), m);
			double[][] db2 = MatrixUtil.divide(dZ2, m);

			// LAYER 1
			double[][] dZ1 = MatrixUtil.multiply(MatrixUtil.dot(MatrixUtil.T(W2), dZ2),
					MatrixUtil.subtract(1.0, MatrixUtil.power(A1, 2)));
			double[][] dW1 = MatrixUtil.divide(MatrixUtil.dot(dZ1, MatrixUtil.T(X)), m);
			double[][] db1 = MatrixUtil.divide(dZ1, m);

			// G.D
			W1 = MatrixUtil.subtract(W1, MatrixUtil.multiply(learningRate, dW1));
			b1 = MatrixUtil.subtract(b1, MatrixUtil.multiply(learningRate, db1));

			W2 = MatrixUtil.subtract(W2, MatrixUtil.multiply(learningRate, dW2));
			b2 = MatrixUtil.subtract(b2, MatrixUtil.multiply(learningRate, db2));

			if (i % 400 == 0 && iprintCost) {
				MatrixUtil.print("==============");
				MatrixUtil.print("Cost = " + cost);
				MatrixUtil.print("Predictions = " + Arrays.deepToString(A2));
				MatrixUtil.print("Real = " + Arrays.deepToString(Y));
				MatrixUtil.print("Predicted precision= "
						+ DNNUtils.accuracyForClassification(A2, Y));
			}
		}
		learnedParameters = new HashMap<>();
		learnedParameters.put(String.valueOf("W_1"), W1);
		learnedParameters.put(String.valueOf("b_1"), b1);
		learnedParameters.put(String.valueOf("W_2"), W2);
		learnedParameters.put(String.valueOf("b_2"), b2);
		return learnedParameters;
	}
/**
 * This is L (deep) layer model for binary classification. 
 * @param X
 * @param Y
 * @param layerDimenstions
 * @param learningRate
 * @param numOfIterations
 * @param optimiser
 * @param iprintCost
 * @return
 * @throws Exception
 */
	public static Map<String, Object> l_LAYER_MODEL_BINARY_CLASSIFICATION(
			double[][] X, double[][] Y, int[] layerDimenstions,
			double learningRate, int numOfIterations,
			DeepNNOptimizers optimiser, boolean iprintCost) throws Exception {

		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);

		//XYSeries //seriesData = new XYSeries("Cost");
		X = MatrixUtil.T(X);
		Y = MatrixUtil.T(Y);

		// System.out.println("Shape of X:" + np.shape(X));
		int m = X[0].length;
		Map<String, Object> parameters = DNNUtils.initializeParameterDeep(
				layerDimenstions, m, ParameterInitializationType.BENGIO);
		Map<Enum<DeepNNKeysForMaps>, Object> AdamParameters = null;
		Map<String, Object> grads = null;
		Map<String, Object> actCaches = null;
		Map<String, Object> feedForwardResult = null;
		double[][] AL = null;
		int L = parameters.size() / 2;
		double cost = 0.0;
		double finalPrecision = 0.0, beta1 = 0.9, beta2 = 0.99, adamt = 2;

		if (optimiser == DeepNNOptimizers.ADAM) {
			AdamParameters = DNNUtils.initializeAdamParameters(parameters);
			beta1 = 0.9;
			beta2 = 0.999;
			adamt = 2;

		}

		for (int i = 0; i < numOfIterations; i++) {

			feedForwardResult = DNNUtils.L_model_forward_binary_classification(
					X, parameters);
			actCaches = (Map<String, Object>) feedForwardResult.get(String
					.valueOf("actCache_" + (L)));
			AL = (double[][]) actCaches.get(String.valueOf("A"));
			cost = DNNUtils.cross_entropy(m, Y, AL);
			//
			grads = DNNUtils.L_model_backward_BinaryClassification(AL, Y,
					feedForwardResult); // here i should pass both
										// cashes;
			if (optimiser == DeepNNOptimizers.ADAM) {
				AdamParameters = DNNUtils.updateParametersWithAdams(parameters,
						grads, AdamParameters, adamt, learningRate, beta1,
						beta2, IPredictionModelConstants.EPSILON);
				parameters = (Map<String, Object>) AdamParameters
						.get(DeepNNKeysForMaps.KEY_FOR_NN_PARAMETERS);
			} else {
				parameters = DNNUtils.updateParametersWithGD(parameters, grads,
						learningRate);
			}

			if (i % 400 == 0 && iprintCost) {
				MatrixUtil.print("==============");
				MatrixUtil.print("Cost = " + cost);
				
				finalPrecision = DNNUtils.accuracyForClassification(AL, Y);
				MatrixUtil.print("Predicted classification precision= "
						+ finalPrecision);
				
			} else if (i % 400 == 0 && !iprintCost) {
				MatrixUtil.print("Cost = " + cost);
			
			}

		}
		
		return parameters;
	}
/**
 * This is L (deep) layer model generic for all the three types of problems i.e. (Rgression, multi-class classification as well as binary classification problems.
 * PS. I have seen if we solve biary class as multiclass classification using this method the accuracy is better.
 * @param X
 * @param Y
 * @param layerDimenstions
 * @param paramInitType
 * @param learningRate
 * @param numOfIterations
 * @param optimiser
 * @param regulizer
 * @param dropOutKeepThreshould
 * @param hiddenLayersActivation
 * @param finalLayerActivation
 * @param costFunction
 * @param totalNumberOfBatches
 * @param isMultiClassClassification
 * @param iprintCost
 * @return
 * @throws Exception
 */
	public static Map<String, Object> l_LAYER_MODEL_Generic(double[][] X,
			double[][] Y, int[] layerDimenstions,
			ParameterInitializationType paramInitType, double learningRate,
			int numOfIterations, DeepNNOptimizers optimiser,
			DeepNNRegularizor regulizer, double dropOutKeepThreshould,
			DeepNNActivations hiddenLayersActivation,
			DeepNNActivations finalLayerActivation,
			DeepNNCostFunctions costFunction, int totalNumberOfBatches,
			boolean isMultiClassClassification, boolean iprintCost)
			throws Exception {

		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);

		//XYSeries //seriesData = new XYSeries("Cost");
		// X = np.T(X);
		// Y = np.T(Y);

		
		int m = X[0].length;

		Map<String, Object> parameters = DNNUtils.initializeParameterDeep(
				layerDimenstions, m, paramInitType);
		Map<Enum<DeepNNKeysForMaps>, Object> AdamParameters = null;
		Map<String, Object> grads = null;
		Map<String, Object> actCaches = null;
		Map<String, Object> feedForwardResult = null;
		double[][] AL = null;
		int L = parameters.size() / 2, costPlotCount = 1;
		double cost = 0.0;
		double finalPrecision = 0.0, beta1 = 0.9, beta2 = 0.99, adamt = 2;
		double[][] X_reamining = X;
		double[][] Y_remaining = Y;
		double[][] X_Single_Batch = null, Y_Batch = null, Y_Single_batch = null;
		Map<String, Object> dropOutMap = new HashMap<>();
		Map<DeepNNKeysForMaps, Object> oneHotMatrixresult = null;
		Map<Enum<DeepNNKeysForMaps>, Object> resultMap = null;
		if (optimiser == DeepNNOptimizers.ADAM) {
			AdamParameters = DNNUtils.initializeAdamParameters(parameters);
			beta1 = 0.9;
			beta2 = 0.999;
			adamt = 2;

		}
		int ithBatchSize = totalNumberOfBatches;
		for (int batch = 1; batch <= totalNumberOfBatches; batch++) {

			double firstBatchSize = 1 - (double) 1 / ithBatchSize--;
			if (batch != totalNumberOfBatches) {
				if (isMultiClassClassification) {
					resultMap = MatrixUtil.splitIntoTrainAndTestForClassification(
							X_reamining, Y_remaining, firstBatchSize);
				} else {
					resultMap = MatrixUtil.splitIntoTrainAndTestForRegression(
							X_reamining, Y_remaining, firstBatchSize);
				}

				X_reamining = (double[][]) resultMap
						.get(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX);
				Y_remaining = (double[][]) resultMap
						.get(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL);
				X_Single_Batch = (double[][]) resultMap
						.get(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX_TEST);
				Y_Batch = (double[][]) resultMap
						.get(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL_TEST);

				if (isMultiClassClassification) {
					oneHotMatrixresult = MatrixUtil.createOneHotVectorMatrix(Y_Batch);
					int numberOfClasses = (int) oneHotMatrixresult
							.get(DeepNNKeysForMaps.KEY_FOR_NUMBER_OF_CLASSES);
					if (numberOfClasses != layerDimenstions[layerDimenstions.length - 1]) {
						System.err
								.println("not enough dimensions sampled, so eskipping the batch with size :"
										+ X_Single_Batch.length);
						continue;
					}
					Y_Single_batch = (double[][]) oneHotMatrixresult
							.get(DeepNNKeysForMaps.KEY_FOR_ONE_HOT_MATRIX);
				} else {

					Y_Single_batch = Y_Batch;
				}

			} else {
				X_Single_Batch = X_reamining;
				if (isMultiClassClassification) {
					oneHotMatrixresult = MatrixUtil
							.createOneHotVectorMatrix(Y_remaining);
					int numberOfClasses = (int) oneHotMatrixresult
							.get(DeepNNKeysForMaps.KEY_FOR_NUMBER_OF_CLASSES);
					Y_Single_batch = (double[][]) oneHotMatrixresult
							.get(DeepNNKeysForMaps.KEY_FOR_ONE_HOT_MATRIX);
					if (numberOfClasses != layerDimenstions[layerDimenstions.length - 1]) {
						System.err
								.println("not enough dimension result sampled, so eskipping the batch with size :"
										+ X_Single_Batch.length);
						continue;
					}
				} else {
					Y_Single_batch = Y_remaining;
				}
			}
			m = X_Single_Batch.length;
			MatrixUtil.print("==============epoch :" + batch + "/ "
					+ totalNumberOfBatches + " with size: "
					+ X_Single_Batch.length);
			X_Single_Batch = MatrixUtil.T(X_Single_Batch);
			Y_Single_batch = MatrixUtil.T(Y_Single_batch);

			for (int i = 0; i < numOfIterations; i++) {
				Stopwatch stopWatch = new Stopwatch();
				feedForwardResult = DNNUtils.L_model_forward_Generic(
						X_Single_Batch, parameters, hiddenLayersActivation,
						finalLayerActivation, regulizer, dropOutKeepThreshould,
						dropOutMap);
				actCaches = (Map<String, Object>) feedForwardResult.get(String
						.valueOf("actCache_" + (L)));
				AL = (double[][]) actCaches.get(String.valueOf("A"));
				// System.out.println("time taking in forward propogation: "+stopWatch.elapsedTime());
				cost = DNNUtils.getCost(m, Y_Single_batch, AL, costFunction);

				//seriesData.add((costPlotCount++), cost);
				//
				// stopWatch=new Stopwatch();
				grads = DNNUtils.L_model_backward_Generic(AL, Y_Single_batch,
						feedForwardResult, hiddenLayersActivation,
						finalLayerActivation, costFunction, regulizer,
						dropOutKeepThreshould, dropOutMap); // here i
															// should
															// pass both
				// System.out.println("time taking in backward propogation: "+stopWatch.elapsedTime());
				// // cashes;
				if (optimiser == DeepNNOptimizers.ADAM) {
					AdamParameters = DNNUtils.updateParametersWithAdams(
							parameters, grads, AdamParameters, adamt,
							learningRate, beta1, beta2,
							IPredictionModelConstants.EPSILON);
					parameters = (Map<String, Object>) AdamParameters
							.get(DeepNNKeysForMaps.KEY_FOR_NN_PARAMETERS);
				} else {
					parameters = DNNUtils.updateParametersWithGD(parameters,
							grads, learningRate);
				}

				if (i % 400 == 0 && iprintCost) {
					MatrixUtil.print("==============");
					MatrixUtil.print("Cost = " + cost);
					// np.print("Predictions = " + Arrays.deepToString(AL));
					// np.print("Real = " + Arrays.deepToString(Y));
					if (isMultiClassClassification) {
						finalPrecision = DNNUtils
								.accuracyForMultiClassClassification(AL,
										Y_Single_batch);
						// np.printshapes(AL, Y_OneHot);
						MatrixUtil.print("Predicted multiclass classification precision= "
								+ finalPrecision);
					} else {
						finalPrecision = DNNUtils.accuracyForClassification(AL,
								Y_Single_batch);
						MatrixUtil.print("Predicted classification precision= "
								+ finalPrecision);
					}

				
				} else if (i % 40 == 0 && !iprintCost) {
					MatrixUtil.print("Cost = " + cost);
					
				}

			}
		}
		
		return parameters;
	}
/**
 * @Todo
 * @param Adj
 * @param X
 * @param Y
 * @param layerDimenstions
 * @param paramInitType
 * @param learningRate
 * @param numOfIterations
 * @param optimiser
 * @param regulizer
 * @param dropOutKeepThreshould
 * @param hiddenLayersActivation
 * @param finalLayerActivation
 * @param costFunction
 * @param totalNumberOfBatches
 * @param isMultiClassClassification
 * @param iprintCost
 * @return
 * @throws Exception
 */
	public static Map<String, Object> l_LAYER_MODEL_Generic_GCN(double[][] Adj,
			double[][] X, double[][] Y, int[] layerDimenstions,
			ParameterInitializationType paramInitType, double learningRate,
			int numOfIterations, DeepNNOptimizers optimiser,
			DeepNNRegularizor regulizer, double dropOutKeepThreshould,
			DeepNNActivations hiddenLayersActivation,
			DeepNNActivations finalLayerActivation,
			DeepNNCostFunctions costFunction, int totalNumberOfBatches,
			boolean isMultiClassClassification, boolean iprintCost)
			throws Exception {return null;}
/**
 * Binary classification with drop outs. Although this classification problem can be implemented using multiclass classification. 
 * But for understanding purpose it is here. Another thing is that it can be use for multilable classificatoin also.
 * @param X
 * @param Y
 * @param layerDimenstions
 * @param learningRate
 * @param numOfIterations
 * @param optimiser
 * @param iprintCost
 * @param dropOutKeepThreshould
 * @return
 * @throws Exception
 */
	public static Map<String, Object> l_LAYER_MODEL_BINARY_CLASSIFICATION_WithDropOut(
			double[][] X, double[][] Y, int[] layerDimenstions,
			double learningRate, int numOfIterations,
			DeepNNOptimizers optimiser, boolean iprintCost,
			double dropOutKeepThreshould) throws Exception {

		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);

		//XYSeries //seriesData = new XYSeries("Cost");
		X = MatrixUtil.T(X);
		Y = MatrixUtil.T(Y);
		// System.out.println("Shape of X:" + np.shape(X));
		int m = X[0].length;
		Map<String, Object> parameters = DNNUtils.initializeParameterDeep(
				layerDimenstions, m, ParameterInitializationType.BENGIO);
		Map<String, Object> grads = null;
		Map<String, Object> actCaches = null;
		Map<String, Object> feedForwardResult = null;
		double[][] AL = null;
		int L = parameters.size() / 2;
		double cost = 0.0;
		double finalPrecision = 0.0, beta1 = 0.9, beta2 = 0.99, adamt = 2;
		Map<Enum<DeepNNKeysForMaps>, Object> AdamParameters = null;
		Map<String, Object> dropOutMap = new HashMap<>();
		if (optimiser == DeepNNOptimizers.ADAM) {
			AdamParameters = DNNUtils.initializeAdamParameters(parameters);
			beta1 = 0.9;
			beta2 = 0.999;
			adamt = 2;

		}

		for (int i = 0; i < numOfIterations; i++) {

			feedForwardResult = DNNUtils
					.L_model_forward_binary_classification_withDropOut(X,
							parameters, dropOutKeepThreshould, dropOutMap);
			actCaches = (Map<String, Object>) feedForwardResult.get(String
					.valueOf("actCache_" + (L)));
			AL = (double[][]) actCaches.get(String.valueOf("A"));
			cost = DNNUtils.cross_entropy(m, Y, AL);
			//seriesData.add((i + 1), cost);
			//
			grads = DNNUtils.L_model_backward_withDropOut(AL, Y,
					feedForwardResult, dropOutKeepThreshould, dropOutMap); // here
																			// i
																			// should
																			// pass
																			// both
																			// cashes;
			if (optimiser == DeepNNOptimizers.ADAM) {
				AdamParameters = DNNUtils.updateParametersWithAdams(parameters,
						grads, AdamParameters, adamt, learningRate, beta1,
						beta2, IPredictionModelConstants.EPSILON);
				parameters = (Map<String, Object>) AdamParameters
						.get(DeepNNKeysForMaps.KEY_FOR_NN_PARAMETERS);
			} else {
				parameters = DNNUtils.updateParametersWithGD(parameters, grads,
						learningRate);
			}

			if (i % 400 == 0 && iprintCost) {
				MatrixUtil.print("==============");
				MatrixUtil.print("Cost = " + cost);
				
				finalPrecision = DNNUtils.accuracyForClassification(AL, Y);
				MatrixUtil.print("Predicted classification precision= "
						+ finalPrecision);
			
			} else if (i % 400 == 0 && !iprintCost) {
				MatrixUtil.print("Cost = " + cost);
				
			}

		}
		
		return parameters;
	}

	public static Map<String, Object> l_LAYER_MODEL_Regression(double[][] X,
			double[][] Y, int[] layerDimenstions, double learningRate,
			int numOfIterations, DeepNNOptimizers optimiser, boolean iprintCost)
			throws Exception {

		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);

		
		X = MatrixUtil.T(X);
		Y = MatrixUtil.T(Y);

		int m = X[0].length;
		Map<String, Object> parameters = DNNUtils.initializeParameterDeep(
				layerDimenstions, m, ParameterInitializationType.HE);
		Map<Enum<DeepNNKeysForMaps>, Object> AdamParameters = null;
		Map<String, Object> grads = null;
		Map<String, Object> actCaches = null;
		Map<String, Object> feedForwardResult = null;
		double[][] AL = null;
		int L = parameters.size() / 2;
		double cost = 0.0;
		double finalPrecision = 0.0, beta1 = 0.9, beta2 = 0.99, adamt = 2;

		if (optimiser == DeepNNOptimizers.ADAM) {
			AdamParameters = DNNUtils.initializeAdamParameters(parameters);
			beta1 = 0.9;
			beta2 = 0.999;
			adamt = 2;

		}

		for (int i = 0; i < numOfIterations; i++) {

			feedForwardResult = DNNUtils.L_model_forward_regression(X,
					parameters);
			actCaches = (Map<String, Object>) feedForwardResult.get(String
					.valueOf("actCache_" + (L)));
			AL = (double[][]) actCaches.get(String.valueOf("A"));
			cost = DNNUtils.cost_mse(m, Y, AL);
			//seriesData.add((i + 1), cost);
			//

			grads = DNNUtils.L_model_backward_regression(AL, Y,
					feedForwardResult); // here i should pass both cashes;

			if (optimiser == DeepNNOptimizers.ADAM) {
				AdamParameters = DNNUtils.updateParametersWithAdams(parameters,
						grads, AdamParameters, adamt, learningRate, beta1,
						beta2, IPredictionModelConstants.EPSILON);
				parameters = (Map<String, Object>) AdamParameters
						.get(DeepNNKeysForMaps.KEY_FOR_NN_PARAMETERS);
			} else {
				parameters = DNNUtils.updateParametersWithGD(parameters, grads,
						learningRate);
			}

			if (i % 400 == 0 && iprintCost) {
				MatrixUtil.print("==============");
				MatrixUtil.print("Cost = " + cost);
				MatrixUtil.print("Predictions = " + Arrays.deepToString(AL));
				MatrixUtil.print("Real = " + Arrays.deepToString(Y));
			
			} else if (i % 400 == 0 && !iprintCost) {
				MatrixUtil.print("Cost = " + cost);
				
			}

		}
		
		return parameters;
	}
/**
 * I was trying to solve ranking problem as a regression problem. @TODO later
 * @param X
 * @param Y
 * @param layerDimenstions
 * @param learningRate
 * @param numOfIterations
 * @param iprintCost
 * @return
 * @throws Exception
 */
	public static Map<String, Object> l_LAYER_MODEL_Ranking(double[][] X,
			double[][] Y, int[] layerDimenstions, double learningRate,
			int numOfIterations, boolean iprintCost) throws Exception {

		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);

		//XYSeries //seriesData = new XYSeries("Cost");
		X = MatrixUtil.T(X);
		Y = MatrixUtil.T(Y);
		// System.out.println("Shape of X:" + np.shape(X));
		int m = X[0].length;
		Map<String, Object> parameters = DNNUtils.initializeParameterDeep(
				layerDimenstions, m, ParameterInitializationType.BENGIO);
		Map<String, Object> grads = null;
		Map<String, Object> actCaches = null;
		Map<String, Object> feedForwardResult = null;
		double[][] AL = null;
		int L = parameters.size() / 2;
		double cost = 0.0;
		double finalPrecision = 0.0;
		for (int i = 0; i < numOfIterations; i++) {

			feedForwardResult = DNNUtils.L_model_forward_ranking(X, parameters);
			actCaches = (Map<String, Object>) feedForwardResult.get(String
					.valueOf("actCache_" + (L)));
			AL = (double[][]) actCaches.get(String.valueOf("A"));
			cost = DNNUtils.sparse_categorial_crossentropy(m, Y, AL);
			//seriesData.add((i + 1), cost);
			//
			grads = DNNUtils.L_model_backward_ranking(AL, Y, feedForwardResult); // here
																					// i
																					// should
																					// pass
																					// both
																					// cashes;
			parameters = DNNUtils.updateParametersWithGD(parameters, grads,
					learningRate);

			if (i % 400 == 0 && iprintCost) {
				MatrixUtil.print("==============");
				MatrixUtil.print("Cost = " + cost);
				MatrixUtil.print("Predictions = " + Arrays.deepToString(AL));
				MatrixUtil.print("Real = " + Arrays.deepToString(Y));
				finalPrecision = DNNUtils.accuracyForClassification(AL, Y);
				MatrixUtil.print("Predicted classification precision= "
						+ finalPrecision);
				
			} else if (i % 400 == 0 && !iprintCost) {
				MatrixUtil.print("Cost = " + cost);
				
			}

		}
		
		return parameters;
	}
/**
 * This is multiclass classification using soft max. Remember we can solve multiclass classification in two ways.
 * 1: using binary classification, oneVsAll method and another 2: using softmax regression. The softmax one is faster obiously.
 * Infact i like softmax for binary classification also. 
 * @param X
 * @param Y
 * @param layerDimenstions
 * @param learningRate
 * @param numOfIterations
 * @param optimizer
 * @param regulizer
 * @param dropOutThreshold
 * @param iprintCost
 * @return
 * @throws Exception
 */
	public static Map<String, Object> l_LAYER_MODEL_MultiClassClassifications_SoftMax(
			double[][] X, double[][] Y, int[] layerDimenstions,
			double learningRate, int numOfIterations,
			DeepNNOptimizers optimizer, DeepNNRegularizor regulizer,
			double dropOutThreshold, boolean iprintCost) throws Exception {

		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);

		
		int m = X[0].length;
		Map<String, Object> parameters = null;
		Map<Enum<DeepNNKeysForMaps>, Object> AdamParameters = null;
		Map<String, Object> grads = null;
		Map<String, Object> actCaches = null;
		Map<String, Object> feedForwardResult = null;
		double[][] AL = null;
		Map<DeepNNKeysForMaps, Object> oneHotMatrixresult = null;
		double cost = 0.0, costDifference = 0.0, previousCost = Double.MIN_VALUE;
		double finalPrecision = 0.0, beta1 = 0.9, beta2 = 0.99, adamt = 2;

		oneHotMatrixresult = MatrixUtil.createOneHotVectorMatrix(Y);
		double[][] Y_OneHot = (double[][]) oneHotMatrixresult
				.get(DeepNNKeysForMaps.KEY_FOR_ONE_HOT_MATRIX);
		Map<Double, double[]> labeltoOnhotVectorMap = (Map<Double, double[]>) oneHotMatrixresult
				.get(DeepNNKeysForMaps.KEY_FOR_LABEL_TO_ONE_HOT);

		X = MatrixUtil.T(X);
		// np.printshapes(X,Y_OneHot);
		Y_OneHot = MatrixUtil.T(Y_OneHot);
		layerDimenstions[layerDimenstions.length - 1] = labeltoOnhotVectorMap
				.size();
		parameters = DNNUtils.initializeParameterDeep(layerDimenstions, m,
				ParameterInitializationType.BENGIO);
		int L = parameters.size() / 2;

		if (optimizer == DeepNNOptimizers.ADAM) {
			AdamParameters = DNNUtils.initializeAdamParameters(parameters);
			beta1 = 0.9;
			beta2 = 0.999;
			adamt = 2;

		}

		for (int i = 0; ((i < numOfIterations)); i++) {

			feedForwardResult = DNNUtils
					.L_model_forward_multiClass_classification(X, parameters);
			actCaches = (Map<String, Object>) feedForwardResult.get(String
					.valueOf("actCache_" + (L)));
			AL = (double[][]) actCaches.get(String.valueOf("A"));
			cost = DNNUtils.sparse_categorial_crossentropy(m, Y_OneHot, AL);
			costDifference = cost - previousCost;
			previousCost = cost;
			// cost = RnnUtils.cross_entropy(m, Y_OneHot, AL);
			//seriesData.add((i + 1), cost);
			//
			grads = DNNUtils.L_model_backward_MultiClass_Classification(AL,
					Y_OneHot, feedForwardResult); // here i
													// should
													// pass both
													// cashes;
			if (optimizer == DeepNNOptimizers.ADAM) {
				AdamParameters = DNNUtils.updateParametersWithAdams(parameters,
						grads, AdamParameters, adamt, learningRate, beta1,
						beta2, IPredictionModelConstants.EPSILON);
				parameters = (Map<String, Object>) AdamParameters
						.get(DeepNNKeysForMaps.KEY_FOR_NN_PARAMETERS);
			} else {
				parameters = DNNUtils.updateParametersWithGD(parameters, grads,
						learningRate);
			}

			if (i % 400 == 0 && iprintCost) {
				learningRate = learningRate / 2;
				MatrixUtil.print("==============");
				MatrixUtil.print("Cost = " + cost);
				
				finalPrecision = DNNUtils.accuracyForMultiClassClassification(
						AL, Y_OneHot);
				// np.printshapes(AL, Y_OneHot);
				MatrixUtil.print("Predicted multiclass classification precision= "
						+ finalPrecision);
			
			}
			if (Math.abs(costDifference) < IPredictionModelConstants.EPSILON) {
				break;
			}

		}
	
		return parameters;
	}

	/**
	 * l_LAYER_MODEL_MultiClassClassifications_SoftMax_withdropout_withBatchSize
	 * Imlementing minibatch for mlticlass classification is tedios task. You can try. Does every min batch should have examples for all the classes?
	 * @param X
	 * @param Y
	 * @param layerDimenstions
	 * @param paramInitType
	 * @param learningRate
	 * @param numOfIterations
	 * @param optimizer
	 * @param regulizer
	 * @param dropOutKeepThreshould
	 * @param totalNumberOfBatches
	 * @param isMultiClassClassification
	 * @param iprintCost
	 * @return
	 * @throws Exception
	 */

	public static Map<String, Object> l_LAYER_MODEL_MultiClassClassifications_SoftMax_withdropout_withBatchSize(
			double[][] X, double[][] Y, int[] layerDimenstions,
			ParameterInitializationType paramInitType, double learningRate,
			int numOfIterations, DeepNNOptimizers optimizer,
			DeepNNRegularizor regulizer, double dropOutKeepThreshould,
			int totalNumberOfBatches, boolean isMultiClassClassification,
			boolean iprintCost) throws Exception {

		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);

		
		int m = X.length, costPlotIteration = 1;
		Map<String, Object> parameters = null;
		Map<Enum<DeepNNKeysForMaps>, Object> AdamParameters = null;
		Map<String, Object> grads = null;
		Map<String, Object> actCaches = null;
		Map<String, Object> feedForwardResult = null;
		double[][] AL = null;
		Map<DeepNNKeysForMaps, Object> oneHotMatrixresult = null;
		double cost = 0.0, previousCost = Double.MIN_VALUE, costDifference = 0.0;
		double finalPrecision = 0.0, beta1 = 0.9, beta2 = 0.99, adamt = 2;
		Map<String, Object> dropOutMap = new HashMap<>();
		
		double[][] X_Single_Batch = null, Y_Batch = null, Y_Single_batch = null;

		Map<Enum<DeepNNKeysForMaps>, Object> resultMap = null;
		
		parameters = DNNUtils.initializeParameterDeep(layerDimenstions, m,
				paramInitType);
		int L = parameters.size() / 2;
		double[][] X_reamining = X;
		double[][] Y_remaining = Y;
		if (optimizer == DeepNNOptimizers.ADAM) {
			AdamParameters = DNNUtils.initializeAdamParameters(parameters);
			beta1 = 0.9;
			beta2 = 0.999;
			adamt = 2;

		}
		// int totalNumberOfBatches=3;
		int ithBatchSize = totalNumberOfBatches;
		for (int batch = 1; batch <= totalNumberOfBatches; batch++) {

			double currentBatchSizePercent = 1 - (double) 1 / ithBatchSize--;
			if (batch != totalNumberOfBatches) {
				if (isMultiClassClassification
						|| (layerDimenstions[layerDimenstions.length - 1] == 2)) {
					resultMap = MatrixUtil.splitIntoTrainAndTestForClassification(
							X_reamining, Y_remaining, currentBatchSizePercent);
				} else {
					resultMap = MatrixUtil.splitIntoTrainAndTestForRegression(
							X_reamining, Y_remaining, currentBatchSizePercent);
				}

				X_reamining = (double[][]) resultMap
						.get(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX);
				Y_remaining = (double[][]) resultMap
						.get(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL);
				X_Single_Batch = (double[][]) resultMap
						.get(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX_TEST);
				Y_Batch = (double[][]) resultMap
						.get(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL_TEST);

				if (isMultiClassClassification) {
					oneHotMatrixresult = MatrixUtil.createOneHotVectorMatrix(Y_Batch);
					int numberOfClasses = (int) oneHotMatrixresult
							.get(DeepNNKeysForMaps.KEY_FOR_NUMBER_OF_CLASSES);
					if (numberOfClasses != layerDimenstions[layerDimenstions.length - 1]) {
						System.err
								.println("not enough dimensions sampled, so eskipping the batch with size :"
										+ X_Single_Batch.length);
						continue;
					}
					Y_Single_batch = (double[][]) oneHotMatrixresult
							.get(DeepNNKeysForMaps.KEY_FOR_ONE_HOT_MATRIX);
				} else {

					Y_Single_batch = Y_Batch;
				}

			} else {
				X_Single_Batch = X_reamining;
				if (isMultiClassClassification) {
					oneHotMatrixresult = MatrixUtil
							.createOneHotVectorMatrix(Y_remaining);
					int numberOfClasses = (int) oneHotMatrixresult
							.get(DeepNNKeysForMaps.KEY_FOR_NUMBER_OF_CLASSES);
					Y_Single_batch = (double[][]) oneHotMatrixresult
							.get(DeepNNKeysForMaps.KEY_FOR_ONE_HOT_MATRIX);
					if (numberOfClasses != layerDimenstions[layerDimenstions.length - 1]) {
						System.err
								.println("not enough dimension result sampled, so eskipping the batch with size :"
										+ X_Single_Batch.length);
						continue;
					}
				} else {
					Y_Single_batch = Y_Batch;
				}
			}
			m = X_Single_Batch.length;
			MatrixUtil.print("==============epoch :" + batch + "/ "
					+ totalNumberOfBatches + " with size: "
					+ X_Single_Batch.length);
			X_Single_Batch = MatrixUtil.T(X_Single_Batch);
			Y_Single_batch = MatrixUtil.T(Y_Single_batch);

			for (int i = 0; (i < numOfIterations); i++) {

				feedForwardResult = DNNUtils
						.L_model_forward_multiClass_classification_withdropout(
								X_Single_Batch, parameters,
								dropOutKeepThreshould, dropOutMap);
				actCaches = (Map<String, Object>) feedForwardResult.get(String
						.valueOf("actCache_" + (L)));
				AL = (double[][]) actCaches.get(String.valueOf("A"));
				cost = DNNUtils.sparse_categorial_crossentropy(m,
						Y_Single_batch, AL);
				costDifference = cost - previousCost;
				previousCost = cost;
				
				grads = DNNUtils
						.L_model_backward_MultiClass_Classification_with_dropout(
								AL, Y_Single_batch, feedForwardResult,
								dropOutKeepThreshould, dropOutMap); // here i
																	// should
																	// pass both
																	// cashes;

				if (optimizer == DeepNNOptimizers.ADAM) {
					AdamParameters = DNNUtils.updateParametersWithAdams(
							parameters, grads, AdamParameters, adamt,
							learningRate, beta1, beta2,
							IPredictionModelConstants.EPSILON);
					parameters = (Map<String, Object>) AdamParameters
							.get(DeepNNKeysForMaps.KEY_FOR_NN_PARAMETERS);
				} else {
					parameters = DNNUtils.updateParametersWithGD(parameters,
							grads, learningRate);
				}

				if (i % 400 == 0 && iprintCost) {
					MatrixUtil.print("==============");
					MatrixUtil.print("Cost = " + cost);
					
					finalPrecision = DNNUtils
							.accuracyForMultiClassClassification(AL,
									Y_Single_batch);
					
					MatrixUtil.print("Predicted multiclass classification precision= "
							+ finalPrecision);
					
				}

			}
		}
		

		return parameters;
	}
/**
 * l_LAYER_MODEL_MultiClassClassifications_SoftMax_withdropout
 * My softmax implementation doesnt produce good result as python (tensorflow one) i dont know why?
 * 
 * @param X
 * @param Y
 * @param layerDimenstions
 * @param learningRate
 * @param numOfIterations
 * @param optimizer
 * @param regulizer
 * @param dropOutKeepThreshould
 * @param iprintCost
 * @return
 * @throws Exception
 */
	public static Map<String, Object> l_LAYER_MODEL_MultiClassClassifications_SoftMax_withdropout(
			double[][] X, double[][] Y, int[] layerDimenstions,
			double learningRate, int numOfIterations,
			DeepNNOptimizers optimizer, DeepNNRegularizor regulizer,
			double dropOutKeepThreshould, boolean iprintCost) throws Exception {

		MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);

		//XYSeries //seriesData = new XYSeries("Cost");

		
		int m = X[0].length;
		Map<String, Object> parameters = null;
		Map<Enum<DeepNNKeysForMaps>, Object> AdamParameters = null;
		Map<String, Object> grads = null;
		Map<String, Object> actCaches = null;
		Map<String, Object> feedForwardResult = null;
		double[][] AL = null;
		Map<DeepNNKeysForMaps, Object> oneHotMatrixresult = null;
		double cost = 0.0, previousCost = Double.MIN_VALUE, costDifference = 0.0;
		double finalPrecision = 0.0, beta1 = 0.9, beta2 = 0.99, adamt = 2;
		Map<String, Object> dropOutMap = new HashMap<>();
		oneHotMatrixresult = MatrixUtil.createOneHotVectorMatrix(Y);
		double[][] Y_OneHot = (double[][]) oneHotMatrixresult
				.get(DeepNNKeysForMaps.KEY_FOR_ONE_HOT_MATRIX);
		Map<Double, double[]> labeltoOnhotVectorMap = (Map<Double, double[]>) oneHotMatrixresult
				.get(DeepNNKeysForMaps.KEY_FOR_LABEL_TO_ONE_HOT);
		X = MatrixUtil.T(X);
		// np.printshapes(X,Y_OneHot);
		Y_OneHot = MatrixUtil.T(Y_OneHot);
		layerDimenstions[layerDimenstions.length - 1] = labeltoOnhotVectorMap
				.size();
		parameters = DNNUtils.initializeParameterDeep(layerDimenstions, m,
				ParameterInitializationType.BENGIO);
		int L = parameters.size() / 2;

		if (optimizer == DeepNNOptimizers.ADAM) {
			AdamParameters = DNNUtils.initializeAdamParameters(parameters);
			beta1 = 0.9;
			beta2 = 0.999;
			adamt = 2;

		}

		for (int i = 0; (i < numOfIterations); i++) {

			feedForwardResult = DNNUtils
					.L_model_forward_multiClass_classification_withdropout(X,
							parameters, dropOutKeepThreshould, dropOutMap);
			actCaches = (Map<String, Object>) feedForwardResult.get(String
					.valueOf("actCache_" + (L)));
			AL = (double[][]) actCaches.get(String.valueOf("A"));
			cost = DNNUtils.sparse_categorial_crossentropy(m, Y_OneHot, AL);
			costDifference = cost - previousCost;
			previousCost = cost;
			
			grads = DNNUtils
					.L_model_backward_MultiClass_Classification_with_dropout(
							AL, Y_OneHot, feedForwardResult,
							dropOutKeepThreshould, dropOutMap); // here i should
																// pass both
																// cashes;

			if (optimizer == DeepNNOptimizers.ADAM) {
				AdamParameters = DNNUtils.updateParametersWithAdams(parameters,
						grads, AdamParameters, adamt, learningRate, beta1,
						beta2, IPredictionModelConstants.EPSILON);
				parameters = (Map<String, Object>) AdamParameters
						.get(DeepNNKeysForMaps.KEY_FOR_NN_PARAMETERS);
			} else {
				parameters = DNNUtils.updateParametersWithGD(parameters, grads,
						learningRate);
			}

			if (i % 400 == 0 && iprintCost) {
				MatrixUtil.print("==============");
				MatrixUtil.print("Cost = " + cost);
				
				finalPrecision = DNNUtils.accuracyForMultiClassClassification(
						AL, Y_OneHot);
				
				MatrixUtil.print("Predicted multiclass classification precision= "
						+ finalPrecision);
				
			}
			if (Math.abs(costDifference) < IPredictionModelConstants.EPSILON) {
				break;
			}

		}
		
		return parameters;
	}
/**
 * This method is basic version for understanding how to solve multiclass classification problem.
 * Second it can be used for multilable classificatoin.
 * @param X
 * @param Y
 * @param layerDimenstions
 * @param learningRate
 * @param numOfIterations
 * @param optimizer
 * @param regulizer
 * @param dropOutThreshold
 * @param iprintCost
 * @return
 * @throws Exception
 */
	public static Map<Double, Map<String, Object>> l_LAYER_MODEL_MultiClassClassifications_OnVsAll(
			double[][] X, double[][] Y, int[] layerDimenstions,
			double learningRate, int numOfIterations,
			DeepNNOptimizers optimizer, DeepNNRegularizor regulizer,
			double dropOutThreshold, boolean iprintCost) throws Exception {

		double[] predictedTrainingLabel = null;
		Set<Double> labels = new HashSet<>();
		Map<Double, Map<String, Object>> learnedParametersFromDNNLists = null;
		Map<String, Object> learnedParameters = null;
		double[][] Y_singleLable = null;
		for (int i = 0; i < Y.length; i++) {
			if (!labels.contains(Y[i][0])) {
				labels.add(Y[i][0]);// / making set of unique lables
			}

		}

		learnedParametersFromDNNLists = new HashMap<Double, Map<String, Object>>();
		for (double label : labels) {
			Y_singleLable = MatrixUtil.createBinaryVector(Y, label);
			MatrixUtil.print("Total labels: " + Arrays.deepToString(labels.toArray())
					+ "learning started for label = " + label);
			
			if (regulizer == DeepNNRegularizor.DROP_OUT) {
				learnedParameters = NeuralNetworkModels
						.l_LAYER_MODEL_BINARY_CLASSIFICATION_WithDropOut(X,
								Y_singleLable, layerDimenstions, learningRate,
								numOfIterations, optimizer, Boolean.TRUE,
								dropOutThreshold);
			} else {
				learnedParameters = NeuralNetworkModels
						.l_LAYER_MODEL_BINARY_CLASSIFICATION(X, Y_singleLable,
								layerDimenstions, learningRate,
								numOfIterations, optimizer, Boolean.TRUE);
			}

			learnedParametersFromDNNLists.put(label, learnedParameters);
		}
		MatrixUtil.print("training accuracy on multiclass = ");
		X = MatrixUtil.T(X);
		predictedTrainingLabel = DNNUtils
				.getPredictedResult_multi_class_classification_one_vs_all(X,
						learnedParametersFromDNNLists, DeepNNActivations.TANH,
						DeepNNActivations.SIGMOID);
		double[] actualLabels = MatrixUtil.toVector(Y);

		
		MatrixUtil.print("accuracy multiclass : "
				+ DNNUtils.accuracyForMultiClassClassification(
						predictedTrainingLabel, actualLabels));
		return learnedParametersFromDNNLists;
	}

}
