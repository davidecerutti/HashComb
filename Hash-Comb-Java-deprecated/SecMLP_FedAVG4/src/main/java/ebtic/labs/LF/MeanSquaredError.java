package ebtic.labs.LF;

import ebtic.labs.utils.Matrix;

/**
 * This class implements Mean Squared Loss function
 *
 */
public class MeanSquaredError extends LossFunction{

	public double exec(double label, double value)
	{
//		System.out.println("("+label+" "+value+")");
		double diff = label - (value+1e-15);
		double squared = Math.pow(2, diff);
		double out = squared;
//		System.out.println("("+diff+" "+squared+" "+out+")");
		return out;
	}

	
	
	public double[][] exec(double[][] labels, double[][] values)
	{
		
		int[] shape = Matrix.getShape(values);
		int row = shape[0];    
        int col = shape[1];
        
        if(row > 1) return null;
        else
        {
        	double[][] output = new double[1][col];
        	
        	for(int j=0; j<col; j++)
        	{
        		double a = labels[0][j];
        		double b = values[0][j];
         		output[0][j] = this.exec(a, b);
        	}
        	
        	return output;
        }
	}  
}
