package com.jnn;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;







import com.phd.models.NeuralNetworkModels;
import com.jnn.consts.*;
import com.jnn.enums.DeepNNActivations;
import com.jnn.enums.DeepNNCostFunctions;
import com.jnn.enums.DeepNNKeysForMaps;
import com.jnn.enums.DeepNNOptimizers;
import com.jnn.enums.DeepNNRegularizor;
import com.jnn.enums.ParameterInitializationType;
import com.jnn.utilities.DNNUtils;
import com.jnn.utilities.MatrixUtil;
import com.jnn.utilities.Stopwatch;

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
public class TestMain {
	public static void main(String[] args) throws Exception {
		String[] files= {"BostonHousePrinceDataset\\\\regressionBostonHousePrince.csv",
				"ionosphereDataset\\ClassificatinoDataSet.txt",
				"heartDisease\\heart.txt",
				"CourseraHandWrittenDigits\\handWrittenDigitsMultiClassClassification.txt",
				
		};
		
	   	String path="..\\data\\"+files[3];//0 for regression, 1,2 Binary classification, 3 for multiclass classificaiton
	//It is supposed that last column in csv file will be the label column
	       double[][] X_train = null;
	       double[][] Y_train = null;
	       double[][] X_test = null;
	       double[][] Y_test = null;	       
	       int m;
	       int numofOfHiddenLayers=3;
	       int nodesInHiddenLayer = 50;
	       double learningRate=0.0075;
	       double trainRatio=0.7;
	       
	       boolean isClassification=true;/// This is needed for random sampling, because for regression problem examples can be sampled uniformly
	       //for training and testing while for classificaiton we need to sample examples from all the classes.
	       Map<Enum<DeepNNKeysForMaps>, Object> resultMap=DNNUtils.readFileAsMatrix(path, ",",Boolean.FALSE,trainRatio,isClassification); 
	       X_train=(double[][])resultMap.get(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX);
	       Y_train=(double[][])resultMap.get(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL);
	       X_test=(double[][])resultMap.get(DeepNNKeysForMaps.KEY_FOR_FEATURE_MATRIX_TEST);
	       Y_test=(double[][])resultMap.get(DeepNNKeysForMaps.KEY_FOR_OUTPUT_LABEL_TEST);

	       m=(int)resultMap.get(DeepNNKeysForMaps.KEY_FOR_NUMBER_EXAMPLES);
	       int noOfBatches=X_train.length/500;
	       if(noOfBatches<1) {
	    	   noOfBatches=1;
	       }
	       int numberOfDimenstions=(int)resultMap.get(DeepNNKeysForMaps.KEY_FOR_NUMBEROFDIMENSIONS);	
	       Map<String, Object> learnedParameters=null;
	       Map<DeepNNKeysForMaps, Object> oneHotMatrixresult=null;
	       X_test=MatrixUtil.T(X_test);
	       Y_test=MatrixUtil.T(Y_test);
	       
	      
	       int numberOfClasses=1;
	       if(isClassification) {
				numberOfClasses=(int)resultMap.get(DeepNNKeysForMaps.KEY_FOR_NUMBER_OF_CLASSES);;	
				}
	       int type=1;//1: Multiclass classification, 2: binary classification , 3: Regression problem
			MatrixUtil.setSeed(IPredictionModelConstants.RANDOM_SEED_FOR_REPRODUCIBILITY);
			MatrixUtil.setGpuEnabled(false);
			int[] layerDimenstions= new int[numofOfHiddenLayers+2];				
			for (int i=1;i<=(numofOfHiddenLayers);i++) {
				layerDimenstions[i]=(int)(nodesInHiddenLayer/( 1));
				if(layerDimenstions[i]<1) {
					layerDimenstions[i]=numberOfDimenstions;
				}
			} 
			layerDimenstions[0]=numberOfDimenstions;
			layerDimenstions[numofOfHiddenLayers+1]=numberOfClasses;
			Map<Double,Map<String, Object>>  multiClassClassificationParameters=null;
			System.out.println("Training started"); 			
			
			if(type==1) {
				Stopwatch stopWatch = new Stopwatch();
				X_train=MatrixUtil.divide(X_train, MatrixUtil.max(X_train));
				X_test=MatrixUtil.divide(X_test, MatrixUtil.max(X_test));
				System.out.println(" time taken: " + stopWatch.elapsedTime()
				
						+ " seconds \n");
						DeepNNActivations hiddenLayerActivations=DeepNNActivations.TANH;
				ParameterInitializationType paramInitType=ParameterInitializationType.XAVIOUR;
			learnedParameters = NeuralNetworkModels.l_LAYER_MODEL_Generic(X_train, Y_train, layerDimenstions,paramInitType, learningRate, 1000, DeepNNOptimizers.ADAM,DeepNNRegularizor.DROP_OUT,0.80,hiddenLayerActivations, DeepNNActivations.SOFTMAX,DeepNNCostFunctions.CATEGORICAL_CROSS_ENTROPY,noOfBatches,true, true);
			
			String dropOut=paramInitType+"_dropout_82_";
			oneHotMatrixresult=MatrixUtil.createOneHotVectorMatrix(MatrixUtil.T(Y_test));
			double[][] Y_OneHot=(double[][])oneHotMatrixresult.get(DeepNNKeysForMaps.KEY_FOR_ONE_HOT_MATRIX);
			Map<Double, double[]> labeltoOnhotVectorMap=(Map<Double, double[]>)oneHotMatrixresult.get(DeepNNKeysForMaps.KEY_FOR_LABEL_TO_ONE_HOT);
			
			double[][] predictedAL=DNNUtils.getPredictedResult_multiclass_classification(X_test, learnedParameters);	
			
			Y_OneHot=MatrixUtil.T(Y_OneHot);
			
			double finalPrecision = DNNUtils.accuracyForMultiClassClassification(predictedAL, Y_OneHot);
			MatrixUtil.print("Predicted multiclass classification on test data precision= " + finalPrecision);
			
			
			MatrixUtil.print("===============================Program finished=========== ");
			
			
			}
			else if (type==2) {
			
			//learnedParameters=NeuralNetworkModels.l_LAYER_MODEL_BINARY_CLASSIFICATION_WithDropOut(X_train, Y_train, layerDimenstions, learningRate, 5000,DeepNNOptimizers.GD, Boolean.TRUE,.95);
			
			 learnedParameters=NeuralNetworkModels.l_LAYER_MODEL_BINARY_CLASSIFICATION(X_train, Y_train, layerDimenstions, learningRate, 5000,DeepNNOptimizers.ADAM, Boolean.TRUE);
			double[][] predictedAL=DNNUtils.getPredictedActivationScoreForGenericActivations(X_test, learnedParameters,DeepNNActivations.TANH, DeepNNActivations.SIGMOID);
			MatrixUtil.print("Predicted precision on test dataset= " + DNNUtils.accuracyForClassification(predictedAL, Y_test));
			
		
			}
			else if (type==3) {
				 learnedParameters=NeuralNetworkModels.l_LAYER_MODEL_Regression(X_train, Y_train, layerDimenstions, learningRate, 10000,DeepNNOptimizers.ADAM, Boolean.TRUE);
					double[][] predictedAL=DNNUtils.getPredictedResult_regression(X_test, learnedParameters);
					MatrixUtil.print("Predictions on Test data= " + Arrays.deepToString(predictedAL));
					MatrixUtil.print("Real on Test data= " + Arrays.deepToString(Y_test));
			}
	       
	}
}
