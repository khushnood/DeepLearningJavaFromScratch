package com.jnn.consts;

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
public final class  IPredictionModelConstants {
    public static final double PI = 3.14159;
    public static final double EPSILON = 1e-8;
    public static final double PLANCK_CONSTANT = 6.62606896e-34;
    public static int NUMBER_OF_EXPERIMENTS_PERFORMED=10;
    public static int ARRAY_LENTH_PER_THREAD=10000;
    public static int MEMORY_SIZE_CUDA=1600000000;
    public static int NUMBER_OF_PARAMTER_ITERATION=1;
    public static final int NUMBER_OF_ITERATION_FOR_OPTIMIZATION=500;
    public static final double OPTIMIZATION_DEFAULT_LEARNING_RATE=0.00001;
    public static final int RANDOM_SEED_FOR_REPRODUCIBILITY=786;
    public static final int TEMP_DEEP_NO_OF_FEATURES=3;
    public static final int TEMP_DEEP_NO_OF_RANDOM_LERNINGTIME=5;
    public static final int GRAPH_NODE_LOW_DIMENSION_SIZE=64;
    public static final int MODEL_INDX_PBP=0;
    public static final int MODEL_INDX_TBP=1;
    public static final int MODEL_INDX_KTBP=2;
    public static final int MODEL_INDX_RBDM=3;
    public static final int MODEL_INDX_PAGE_RANK=4;
    public static final int MODEL_INDX_PAGE_RANK_HYB=5;
    public static final int MODEL_INDX_INDEGREE=6;
    public static final int MODEL_INDX_NON_PARAM=7;
    public static final int MODEL_INDX_NEW_TEST=8;
    public static final int MODEL_INDX_POISSON_PROCESS=9;
    public static final int MODEL_INDX_POISSON_PROCESS_TEST=10;
    public static final int MODEL_INDX_NEW_TBP_NONPARAM=11;
    public static final int MODEL_INDX_HAWKES_PROCESS=12;
    public static final int MODEL_INDX_TEMPG_DNN=13;
    public static final int MODEL_INDX_TEMPG_GCNN=14;
    public static String[] kernalNames= {"exponential","linear"};
    public static  String[] MODEL_NAME_LABLE = {"PBP","TBP","KTBP","RBDM","PR","PRHYB","INDG","_NON_PARAM","_UNKNOWM","POISSON_PROCESS","POISSON_TEST","TBP_NONPARAM","HAWKES","TEMPG_DNN","DNN_N2VEC" };
    public static String[] fileLabelNames = {"_MOVIELENS_10M","_MOVIELENS_20M", "_NETFLIX", "_FACEBOOK","_YOUCHOOSE","_AMAZON","_CITATION","_HUMANCONTACT","_STACKOVERFLOW","_YOUTUBE","_DIGG","_INFECTIOUS","_RT","_EmailEUCore","_APS_CITATION","_UNKNOWN" };
    public static final int NUMBER_OF_INDECES_RESULT_FILE=23;
    public static final int FUTURE_TIME_LENGHT=30;
    public static final int PAST_TIME_LENGHT=30;
    public static  String[] FILE_NAMES = {
		"movielens_10M.txt","movielens.txt",
		"netflix5k_result_time_desc.txt", "facebook.txt",
		"YouChooseTimeAsDayNormalized.txt", "amazonTBPNormalizedTime.txt",
		"citationArxivTimeAsMonth.txt", "HumunContacMIT.txt",
		"stackOverflowFavNormalizedTime.txt",
		"youTubeTemporalNormalized.txt", "diggDataSet.txt","infectiousDisease.txt","retweet.txt","emailEUCore.txt","apsCitationProcessed.txt" };
    public static int FILE_NAME_INDX_MOVIELENS=0;
    public static int FILE_NAME_INDX_NETFLIX=1;
    public static int FILE_NAME_INDX_FACEBOOK=2;
    public static int FILE_NAME_INDX_YOUCHOOSE=3;
    public static int FILE_NAME_INDX_AMAZON=4;
    public static int FILE_NAME_INDX_CITATION=5;
    public static int FILE_NAME_INDX_HUMANCONTACT_MIT=6;
    public static int FILE_NAME_INDX_STACK_OVERFLOW=7;
    public static int FILE_NAME_INDX_YOUTUBE=8;
    public static int FILE_NAME_INDX_DIGG=9;
    public static int FILE_NAME_INDX_RE_TWEET=12;
    public static int FILE_NAME_INDX_UKNOWN=11;
    public  static String OS = System.getProperty("os.name").toLowerCase();
    
    public static double  DT=0.01;      // Time Step for solving integral eq. (hour)
    public	static double Th=-0.242;     // Theta: Kernel parameter
    public	static int  WIN = 4 ;     // Window for estimating p_T
    public	static int  DIM = 4 ;     // # parameters: p(t)
}