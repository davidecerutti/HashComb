package ebtic.labs.AF;

import java.util.Arrays;

import ebtic.labs.NN.NN;
import ebtic.labs.utils.Matrix;

/**
 * This class implements Sigmoid Activation function and the its derivative
 * 
 */
public class SoftMAX extends ActivationFunction{

	
	
	public static int[] softmaxClassAssignment(double[][] input) {
        int[] classAssignments = new int[input.length];  // To store the class assignments

        for (int i = 0; i < input.length; i++) {
            double[] softmaxRow = softmax(input[i]);
            classAssignments[i] = argMax(softmaxRow);  // Get the index of the max value (class with highest probability)
        }

        return classAssignments;
    }
	
	
	// Method to find the index of the maximum value in an array (argMax)
    public static int argMax(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
	
	
    public static double[] softmax(double[] inputRow) {
        double[] softmaxOutput = new double[inputRow.length];

        // Step 1: Calculate the exponentials and sum them up
        double max = Arrays.stream(inputRow).max().getAsDouble();  // To avoid overflow
        double sumExp = 0.0;
        for (int i = 0; i < inputRow.length; i++) {
            softmaxOutput[i] = Math.exp(inputRow[i] - max);  // Subtract max for numerical stability
            sumExp += softmaxOutput[i];
        }

        // Step 2: Divide each exponential by the sum of exponentials
        for (int i = 0; i < inputRow.length; i++) {
            softmaxOutput[i] /= sumExp;
        }

        return softmaxOutput;
    }

	
	@Override
	public double[][] exec(double[][] Z) {
		int[] shape = Matrix.getShape(Z);
		int row = shape[0];    
        int col = shape[1];
        double[][] output = new double[row][col];
		for(int i=0; i<row; i++)
		{
			
			output[i] = softmax(Z[i]);
			
				
		}
		return output;

	}

	@Override
	public double[][] exec_derivative(double[][] A) {
		
		return null;
	}


	@Override
	public Types getType() {
		// TODO Auto-generated method stub
		return Types.SoftMAX;
	}

}
