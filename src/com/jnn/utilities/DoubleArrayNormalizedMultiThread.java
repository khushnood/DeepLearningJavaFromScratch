package com.jnn.utilities;
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
public class DoubleArrayNormalizedMultiThread extends Thread{


	private double[] arr;

	private int low, high; 
	double partial;
	double divisor;
	

	public DoubleArrayNormalizedMultiThread(double[] arr, int low, int high,double divisor) {
		this.arr = arr;
		this.low = low;
		this.high = Math.min(high, arr.length);
		this.divisor=divisor;
	}
	
	public double getPartialSum() {
		return partial;
	}

	public void run() {
		partial = divide(arr, low, high,divisor);
	}

	public static double normalized(double[] arr,double divisor) {
		return divide(arr, 0, arr.length,divisor);
	}

	public static double divide(double[] arr, int low, int high,double divisor) {
		double total = 0;

		for (int i = low; i < high; i++) {
			arr[i]=arr[i]/divisor;
		}

		return total;
	}

	public static double[] parellelDivision(double[] arr,double divisor) {
		return parallelDivision(arr, Runtime.getRuntime().availableProcessors(),divisor);
	}

	public static double[] parallelDivision(double[] arr, int threads,double divisor) {
		int size = (int) Math.ceil(arr.length * 1.0 / threads);

		DoubleArrayNormalizedMultiThread[] sums = new DoubleArrayNormalizedMultiThread[threads];

		for (int i = 0; i < threads; i++) {
			sums[i] = new DoubleArrayNormalizedMultiThread(arr, i * size, (i + 1) * size,divisor);
			sums[i].start();
		}

		try {
			for (DoubleArrayNormalizedMultiThread sum : sums) {
				sum.join();
			}
		} catch (InterruptedException e) {
		}

		

		return arr;
	}
	


}
