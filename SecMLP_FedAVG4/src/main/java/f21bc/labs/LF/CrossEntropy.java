package f21bc.labs.LF;

import f21bc.labs.utils.Matrix;

/**
 * This class implements Cross Entropy Loss function
 *
 */
public class CrossEntropy extends LossFunction{

	
	public double exec(double label, double value)
	{
//		System.out.println("("+label+" "+value+")");
		double a = label*Math.log(value+1e-15);
		double b = (1-label)*Math.log(1-value+1e-15);
		double out = -1*(a+b);
//		System.out.println("("+a+" "+b+" "+out+")");
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
