package f21bc.labs.AF;

import f21bc.labs.NN.NN;
import f21bc.labs.utils.Matrix;

/**
 * This class implements RelU Activation function and the its derivative
 * 
 */
public class ReLU extends ActivationFunction{

	private double[][] Z = null;
	
	private double reLU(double z)
	{
//		System.out.println("Weight "+z);
		if(z>=0)
			return z;
		else
			return 0;
			
				
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
				output[i][j] = reLU(Z[i][j]);
				
		}
		
		this.Z = Z;
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
			{
				double aux = this.Z[i][j];
				if(aux>=0)
					output[i][j]= 1;
				else
					output[i][j]= 0;
			}
				
				
		}
		return output;
	}

	@Override
	public Types getType() {
		// TODO Auto-generated method stub
		return Types.ReLU;
	}

}
