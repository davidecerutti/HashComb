package ebtic.labs.AF;

import ebtic.labs.NN.NN;
import ebtic.labs.utils.Matrix;

/**
 * This class implements Sigmoid Activation function and the its derivative
 * 
 */
public class Sigmoid extends ActivationFunction{

	
	
	public double sigmoid(double z)
	{
		return(1/(1 + Math.pow(Math.E, (-1*z))));
	}
	
	@Override
	public double[][] exec(double[][] Z) {
		int[] shape = Matrix.getShape(Z);
		int row = shape[0];    
        int col = shape[1];
        double[][] output = new double[row][col];
		for(int i=0; i<row; i++)
		{
			for(int j=0; j<col; j++)
				output[i][j] = sigmoid(Z[i][j]);
				
		}
		return output;
	}

	@Override
	public double[][] exec_derivative(double[][] A) {
		
		int[] shape = Matrix.getShape(A);
		int row = shape[0];    
        int col = shape[1];
        double[][] output = new double[row][col];
		for(int i=0; i<row; i++)
		{
			for(int j=0; j<col; j++)
				output[i][j] = A[i][j] * (1 - A[i][j]);
				
		}
		return output;
	}

	@Override
	public Types getType() {
		// TODO Auto-generated method stub
		return Types.SIGMOID;
	}

}
