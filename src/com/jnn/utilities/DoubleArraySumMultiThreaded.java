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
public class DoubleArraySumMultiThreaded extends Thread {

	private double[] arr;

	private int low, high; 
	double partial;
	

	public DoubleArraySumMultiThreaded(double[] arr, int low, int high) {
		this.arr = arr;
		this.low = low;
		this.high = Math.min(high, arr.length);
	}
	
	public double getPartialSum() {
		return partial;
	}

	public void run() {
		partial = sum(arr, low, high);
	}

	public static double sum(double[] arr) {
		return sum(arr, 0, arr.length);
	}

	public static double sum(double[] arr, int low, int high) {
		double total = 0;

		for (int i = low; i < high; i++) {
			total += arr[i];
		}

		return total;
	}

	public static double parallelSum(double[] arr) {
		return parallelSum(arr, Runtime.getRuntime().availableProcessors());
	}

	public static double parallelSum(double[] arr, int threads) {
		int size = (int) Math.ceil(arr.length * 1.0 / threads);

		DoubleArraySumMultiThreaded[] sums = new DoubleArraySumMultiThreaded[threads];

		for (int i = 0; i < threads; i++) {
			sums[i] = new DoubleArraySumMultiThreaded(arr, i * size, (i + 1) * size);
			sums[i].start();
		}

		try {
			for (DoubleArraySumMultiThreaded sum : sums) {
				sum.join();
			}
		} catch (InterruptedException e) {
		}

		double total = 0;

		for (DoubleArraySumMultiThreaded sum : sums) {
			total += sum.getPartialSum();
		}

		return total;
	}
	

}