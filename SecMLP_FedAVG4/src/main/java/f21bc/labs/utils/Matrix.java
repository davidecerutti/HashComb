package f21bc.labs.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import f21bc.labs.Exceptions.MatrixException;
import f21bc.labs.Federated.Noise.GaussianNoise;
import f21bc.labs.objects.MyObject;
/**
 * The class implements all the mathematical oerations involving vectors and MAtrix that could be possible used in the training
 * @author maurizio
 *
 */
public class Matrix {
	
	
	/**
	 * Return a matrix of the values contained in the input file
	 * @param objects each element in the list is the representation of a line in the input file (a sample)
	 * @return the matrix of the input
	 */
	public static double[][] getData(List<MyObject> objects)
	{
		double[][] rows = new double[objects.size()][];
		for(int i=0; i<objects.size(); i++)
			{
				rows[i] = objects.get(i).returnData() ;
				
			}
			
		return rows;
		
	}
	
	/**
	 * Return the labels of the values contained in the input file
	 * @param objects
	 * @return
	 */
	public static double[][] getLabels(List<MyObject> objects)
	{
		double[][] rows = new double[1][objects.size()];
		
		for(int i=0; i<objects.size(); i++)
			{
			
				rows[0][i] = objects.get(i).returnLabel() ;
				
			}
		
		return rows;
		
	}
	
	
	/**
	 * The method implement the clipping of the content of the matrix
	 * @param m input matrix
	 * @param clip input clipping value
	 * @return the matrix after clipping 
	 */
		public static double[][] clip(double[][] m, double clip)
		{
			double cP = Math.abs(clip);
			double cN = cP*(-1);
			for(int i=0; i<m.length; i++)
			{
				double[] aux = m[i];
				for(int j=0; j<aux.length; j++)
				{
					double value = m[i][j];
					if(value>cP)
						m[i][j] = cP;
					else if(value<cN)
						m[i][j] = cN;
						
				}

			}
			
			return m;
		}

	
	
/**
 * The method implement the representaton of the content of the matrix
 * @param m input matrix
 * @return the string representing the input
 */
	public static String toString(double[][] m)
	{
		String out = "[ \n";
		String out2 = "";
		for(int i=0; i<m.length; i++)
		{
			out2 = out2 + "[";
			double[] aux = m[i];
			for(int j=0; j<aux.length; j++)
				out2 = out2 +" "+aux[j];
			out2 = out2 + " ]\n";
		}
		
		return out+out2+" ]";
	}
	
	/**
	 * The method implement the representaton of the content of the matrix
	 * @param m input matrix
	 * @return the string representing the input
	 */
		public static String toString(double[] b)
		{
			String out = "[ ";
			for(int i=0; i<b.length; i++)
			{
				
					out = out +", "+b[i];
				
			}
			
			return out+" ]";
		}
	
	
	/**
	 * The method returns the shape of the matrix in input
	 * @param o matrix 
	 * @return shape as an array of integer [2]
	 */
	public static int[] getShape(double[][] o)
	{
		int row = o.length;
		int col = o[0].length;

		return new int[]{row,col};
	}
	
	
	
	/**
	 * The method returns the shape of the matrix in input for a generic object
	 * @param o matrix 
	 * @return shape as an array of integer [2]
	 */
	public static int[] getShape(Object[][] o)
	{
		int row = o.length;
		int col = o[0].length;

		return new int[]{row,col};
	}
	

	public static String printShape(Object[][] o)
	{
		int row = o.length;
		int col = o[0].length;

		int[] out = new int[]{row,col};
		
		return "("+out[0]+")x("+out[1]+")";
	}
	
	public static String printShape(double[][] o)
	{
		int row = o.length;
		int col = o[0].length;

		int[] out = new int[]{row,col};
		
		return "("+out[0]+")x("+out[1]+")";
	}
	
	/**
	 * Calculate the transpose of the matrix
	 * @param matrix in input
	 * @return transpose matrix
	 */
	public static double[][] transpose(double[][] matrix)
	{ 
		int[] shape = Matrix.getShape(matrix);  
		int row = shape[0];
		int column = shape[1];
		
		double[][] transpose = new double[column][row];
        for(int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++) {
                transpose[j][i] = matrix[i][j];
            }
        }
        
        return transpose;
	}
	
	
	/**
	 * The method extract a sub matrix 
	 * @param matrix in input of dimension N*M
	 * @param beg index where to start 
	 * @param end index where to stop
	 * @return a sub matrix of the original, dimension n*M where n<=N
	 */
	 public static double[][] subMatrix(double[][] matrix, int beg, int end) {
	        return Arrays.copyOfRange(matrix, beg, end);
	        
	    }
	
	 
	 public static double[][] subVector(double[][] matrix, int beg, int end) {
	        //return Arrays.copyOfRange(matrix, beg, end);
		 double[][] yT = Matrix.transpose(matrix);
		 return Matrix.transpose(Matrix.subMatrix(yT, beg, end));
	    }

	 

	/**
	 * Implements the sum of 2 matrix of the same shape N*M or where at least one is 1*M, N*1
	 * @param A
	 * @param B
	 * @return a matrix of shape N*M
	 * @throws MatrixException
	 */
	public static double[][] sum(double[][] A, double[][] B ) throws MatrixException
	{
		int[] shapeA = Matrix.getShape(A);
		int[] shapeB = Matrix.getShape(B);
		double[][] output = null;
		
		String error = "operands could not be broadcast together with shapes ("+shapeA[0]+","+shapeA[1]+") ("+shapeB[0]+","+shapeB[1]+")";
		
				
		// same shape
		if((shapeA[0]==shapeB[0])&&(shapeA[1]==shapeB[1]))
		{
			output = new double[shapeA[0]][shapeB[1]];
			
			for(int i=0; i<shapeA[0]; i++)
			{
				for(int j=0; j<shapeA[1]; j++)
					output[i][j] = A[i][j] + B[i][j];
			}
			
			
			return output;
		}
		
		// one is an array (orizzontal)
		else if((shapeA[0]==1)||(shapeB[0]==1))
		{
			if(shapeA[0]==1)
			{
				
				//only one element
				if(shapeA[1]==1)
				{
					output = new double[shapeB[0]][shapeB[1]];
					for(int i=0; i<shapeB[0]; i++)
					{
						for(int j=0; j<shapeB[1]; j++)
							output[i][j] = A[0][0] + B[i][j];
					}
					return output;
				}
				//same column
				if(shapeA[1]==shapeB[1])
				{
					output = new double[shapeB[0]][shapeB[1]];
					for(int i=0; i<shapeB[0]; i++)
					{
						for(int j=0; j<shapeA[1]; j++)
							output[i][j] = A[0][j] + B[i][j];
					}
					return output;
				}
				// both arrays
				else if(shapeB[1]==1)
				{
					output = new double[shapeB[0]][shapeA[1]];
					for(int j=0; j<shapeA[1]; j++)
					{
						for(int i=0; i<shapeB[0]; i++)
							output[i][j] = A[0][j] + B[i][0];
					}
					return output;
				}
				
				else
				{
					throw new MatrixException(error);
				}
			}
			
			
			else if(shapeB[0]==1)
			{
				
				//only one element
				if(shapeB[1]==1)
				{
					output = new double[shapeA[0]][shapeA[1]];
					for(int i=0; i<shapeA[0]; i++)
					{
						for(int j=0; j<shapeA[1]; j++)
							output[i][j] = B[0][0] + A[i][j];
					}
					return output;
				}
				//same column
				if(shapeB[1]==shapeA[1])
				{
					output = new double[shapeA[0]][shapeA[1]];
					for(int i=0; i<shapeA[0]; i++)
					{
						for(int j=0; j<shapeB[1]; j++)
							output[i][j] = B[0][j] + A[i][j];
					}
					return output;
				}
				// both arrays
				else if(shapeA[1]==1)
				{
					output = new double[shapeA[0]][shapeB[1]];
					for(int j=0; j<shapeB[1]; j++)
					{
						for(int i=0; i<shapeA[0]; i++)
							output[i][j] = B[0][j] + A[i][0];
					}
					return output;
				}
				
				else
				{
					throw new MatrixException(error);
				}
			}
			
			
		}
		
		else if((shapeA[1]==1)||(shapeB[1]==1))
		{
			
			if(shapeA[1]==1)
			{
				//same column
				if(shapeA[0]==shapeB[0])
				{
					output = new double[shapeB[0]][shapeB[1]];
					for(int i=0; i<shapeB[0]; i++)
					{
						for(int j=0; j<shapeB[1]; j++)
							output[i][j] = A[i][0] + B[i][j];
					}
					return output;
				}
				// both arrays 
				else if(shapeB[0]==1)
				{
					output = new double[shapeA[0]][shapeB[1]];
					for(int i=0; i<shapeA[0]; i++)
					{
						for(int j=0; j<shapeB[1]; j++)
							output[i][j] = A[i][0] + B[0][j];
					}
					return output;
				}
				
				else
				{
					throw new MatrixException(error);
				}
			}
			
			if(shapeB[1]==1)
			{
				//same column
				if(shapeA[0]==shapeB[0])
				{
					output = new double[shapeA[0]][shapeA[1]];
					for(int i=0; i<shapeA[0]; i++)
					{
						for(int j=0; j<shapeA[1]; j++)
							output[i][j] = B[i][0] + A[i][j];
					}
					return output;
				}
				// both arrays 
				else if(shapeA[0]==1)
				{
					output = new double[shapeB[0]][shapeA[1]];
					for(int i=0; i<shapeB[0]; i++)
					{
						for(int j=0; j<shapeA[1]; j++)
							output[i][j] = B[i][0] + A[0][j];
					}
					return output;
				}
				
				else
				{
					throw new MatrixException(error);
				}
			}
			
			
			
		}
		
		else
		{
			throw new MatrixException(error);
		}
		
		return null;
		
	}

	/**
	 * The method implements the subbtraction between matrix, the same rules of sum apply
	 * @param A
	 * @param B
	 * @return
	 * @throws MatrixException
	 */
	public static double[][] sub(double[][] A, double[][] B ) throws MatrixException
	{
		// first calculate the negative of B
		double[][] aux = Matrix.neg(B);
		// and then sum A and B
		return Matrix.sum(A, aux);
	}
	

	
	public static double[][] divide(double[][] A, double b)
	{
		
		//Calculates number of rows and columns present in first matrix    
        int[] shapeA = Matrix.getShape(A);
		int row = shapeA[0];    
        int col = shapeA[1]; 
        
        double[][] prod = new double[row][col];
        
        for(int i = 0; i < row; i++)
        {
        	for(int j = 0; j < col; j++)
            {
        		prod[i][j] = A[i][j]/b;
//        		System.out.println(A[i][j]+" / "+b+"  --> "+prod[i][j]);
            }
        }
		
        return prod;
	}
	
	public static double[][] product(double b, double[][] A)
	{
		
		//Calculates number of rows and columns present in first matrix    
        int[] shapeA = Matrix.getShape(A);
		int row = shapeA[0];    
        int col = shapeA[1]; 
        
        double[][] prod = new double[row][col];
        
        for(int i = 0; i < row; i++)
        {
        	for(int j = 0; j < col; j++)
            {
        		prod[i][j] = b * A[i][j];
            }
        }
		
        return prod;
	}
	
	
	
	public static double[][] product(double[][] A, double[][] B)
	{
		
		double prod[][] = null;
		//Calculates number of rows and columns present in first matrix    
        int[] shapeA = Matrix.getShape(A);
		int row1 = shapeA[0];    
        int col1 = shapeA[1];    
        
        
      //Calculates the number of rows and columns present in the second matrix    
  
        int[] shapeB = Matrix.getShape(B);
		int row2 = shapeB[0];    
        int col2 = shapeB[1];        
      
      //For two matrices to be multiplied,     
      //number of columns in first matrix must be equal to number of rows in second matrix    
      
      if(col1 != row2){    
          System.out.println("Matrices cannot be multiplied");    
      }    
      
      else{    
          //Array prod will hold the result    
          prod = new double[row1][col2];    
              
          //Performs product of matrices a and b. Store the result in matrix prod    
          for(int i = 0; i < row1; i++){    
              for(int j = 0; j < col2; j++){    
                  for(int k = 0; k < row2; k++){
                	 //System.out.println(prod[i][j]+"  "+A[i][k]+" * "+B[k][j]); 
                     prod[i][j] = prod[i][j] + A[i][k] * B[k][j];     
                  }    
              }    
          }    
              
//          System.out.println("Product of two matrices: ");    
//          for(int i = 0; i < row1; i++){    
//              for(int j = 0; j < col2; j++){    
//                 System.out.print(prod[i][j] + " ");    
//              }    
//              //System.out.println();    
//          }    
      }  
		
		return prod;
	}
	

	
	/**
	 * The method implements the Hadamard product of matrix with same shape M*N
	 * @param A
	 * @param B
	 * @return a matrix of the same shape as A and B in which each element (i,j) is the product of the correspondant elements in the 2 matrix 
	 * @throws MatrixException
	 */
	public static double[][] hadamard(double[][] A, double[][] B ) throws MatrixException
	{ 
		int[] shapeA = Matrix.getShape(A);
		int[] shapeB = Matrix.getShape(B);
		double[][] output = null;
		
		String error = "operands could not be broadcast together with shapes ("+shapeA[0]+","+shapeA[1]+") ("+shapeB[0]+","+shapeB[1]+")";
		
		if((shapeA[0]!=shapeB[0])||(shapeA[1]!=shapeB[1]))
			
			throw new MatrixException(error);
		
		else 
		{
			output = new double[shapeA[0]][shapeB[1]];
			
			for(int i=0; i<shapeA[0]; i++)
			{
				for(int j=0; j<shapeA[1]; j++)
					output[i][j] = A[i][j] * B[i][j];
			}
			
			
			return output;
		}
		
		
		
		
	}

	
	/**
	 * The method implement the negative matrix
	 * @param A input matrix
	 * @return negative of the input
	 */
	public static double[][] neg(double[][] A)
	{
		int[] shapeA = Matrix.getShape(A);
		double[][] output = new double[shapeA[0]][shapeA[1]];
		
		
		for(int i =0; i< shapeA[0]; i++)
		{
			for(int j=0; j<shapeA[1]; j++)
				output[i][j] = (-1) * A[i][j];
		}
		
		return output;
	}

	
	/**
	 * The method implement the sum by row of a matrix with shape N*M
	 * @param A input matrix
	 * @return a matrix of shape N*1 in which each element is the dum of the M elements in the correspondent row of A 
	 */
	public static double[][] sum_by_Row(double[][] A)
	{
		int[] shapeA = Matrix.getShape(A);
		double[][] output = new double[shapeA[0]][1];
		
		
		for(int i =0; i< shapeA[0]; i++)
		{
			double cell = 0;
			for(int j=0; j<shapeA[1]; j++)
				cell = cell + A[i][j];
			output[i][0] = cell;
		}
		
		return output;
	}

	/**
	 * The method implement the sum by column of a matrix with shape N*M
	 * @param A input matrix
	 * @return a matrix of shape M*1 in which each element is the sum of the n elements in the correspondent column of A 
	 */
	public static double[][] sum_by_Col(double[][] A)
	{
		int[] shapeA = Matrix.getShape(A);
		double[][] output = new double[1][shapeA[1]];
		
		
		for(int j =0; j< shapeA[1]; j++)
		{
			double cell = 0;
			for(int i=0; i<shapeA[0]; i++)
				cell = cell + A[i][j];
			output[0][j] = cell;
		}
		
		return output;
	}
	
	/**
	 * The method calculate the average values by row of A, with shape N*M
	 * @param A a matrix
	 * @return a matrix of shape N*1 in which each element is the average of the element of the M columns for the same row
	 */
	public static double[][] average_by_row(double[][] A)
	{
		int[] shapeA = Matrix.getShape(A);
		double aux = 1.0 / shapeA[1]; 
		double[][] output = sum_by_Row(A);

		output = product(aux, output);
		
		return output;
	}
	
	
	/**
	 * The method generate a matrix with fixed values
	 * @param rows number of rows
	 * @param columns number of columns
	 * @param value the fix value, each element of the new matrix is initialized with this
	 * @return the new matrix
	 */
	public static double[][] full(int rows, int columns, double value)
	{
		double[][] output = new double[rows][columns];
		
		for(int i=0; i<rows; i++)
		{
			for(int j=0; j<columns; j++)
			{
				output[i][j] = value;
			}
		}
		
		return output;
	}
	
	
	public static double[][] full(int rows, int columns, double min, double max)
	{
		double[][] output = new double[rows][columns];
		
		double high = rows * columns;
		double low = max - min;
		
		for(int i=0; i<rows; i++)
		{
			int a = i;
			if(i==0)
				a=1;
			
			for(int j=0; j<columns; j++)
			{
				int b = j;
				if(b==0)
					b=1;
					
				double z = a*b;
				//sig%(max-min+1)+min
				output[i][j] = ((z * low)/high) + min;
//				System.out.println("values: "+i+" "+j+"  "+z+"   "+output[i][j]+"  "+low);
			}
		}
		
//		System.out.println(Matrix.toString(output));
		return output;
	}
     
	/**
	 * The method generate a new matrix with random values within a range 
	 * @param rows the number of rows for the matrix
	 * @param columns the number of columns for the new matrix
	 * @param min the min value in range
	 * @param max the max value in range
	 * @return the new matrix
	 */
    public static double[][] random(int rows, int columns, double min, double max)
	{
    	
		double[][] output = new double[rows][columns];
		
		for(int i=0; i<rows; i++)
		{
			for(int j=0; j<columns; j++)
			{
				double x = new Random().doubles(min, max).limit(1).findFirst().getAsDouble();
				
				output[i][j] = x;
			}
		}
		
		return output;
	}

    
    public static double[][] xavier(int rows, int columns)
 	{
     	
 		double[][] output = new double[rows][columns];
 		double aux = ((double)6 / (double)(rows));
 		double limit = Math.sqrt(aux);
 		Random random = new Random();
 		
 		for(int i=0; i<rows; i++)
 		{
 			for(int j=0; j<columns; j++)
 			{
 				double x = random.nextDouble() * (2 * limit) - limit;
 				
 				output[i][j] = x;
 			}
 		}
// 		System.out.println(Matrix.toString(output));
 		return output;
 	}
    
    
    public static double[][] MSRA(int rows, int columns)
  	{
      	
  		double[][] output = new double[rows][columns];
  		double aux = ((double)2 / (double)(rows));
  		double limit = Math.sqrt(aux);
  		Random random = new Random();
  		
  		for(int i=0; i<rows; i++)
  		{
  			for(int j=0; j<columns; j++)
  			{
  				double x = random.nextGaussian() * limit;
  				
  				output[i][j] = x;
  			}
  		}
 // 		System.out.println(Matrix.toString(output));
  		return output;
  	}
    
    /**
	 * The method generate a new matrix with random values within the range [0, 1] 
	 * @param rows the number of rows for the matrix
	 * @param columns the number of columns for the new matrix
	 * @return the new matrix
	 */
	public static double[][] random(int rows, int columns)
	{
		double[][] output = new double[rows][columns];

		for(int i=0; i<rows; i++)
		{
			for(int j=0; j<columns; j++)
			{
				
				output[i][j] = Math.random();
			}
		}
		
		return output;
	}
	
	
	/**
	 * The method concats 2 matrix of shape N*M and Q*M
	 * @param a
	 * @param b
	 * @return a matrix resulting from the concatenation of the input
	 */
	public static double[][] concatMatrix(double[][] a, double[][] b) {
		int aLen = a.length;
        int bLen = b.length;
        double[][] result = new double[aLen + bLen][a[0].length];

        System.arraycopy(a, 0, result, 0, aLen);
        System.arraycopy(b, 0, result, aLen, bLen);
        
        return result;
	}

	/**
	 * The class iterate through the input matrix and swaps the indicated values
	 * @param A input matrix
	 * @param origValue the value to look for
	 * @param newValue the new value 
	 * @return a matrix of the same dimensions as A with the value 'origValue' swapped with 'newValue'
	 */
	public static double[][] swapValues(double[][] A, double origValue, double newValue)
	{
		int[] shapeA = Matrix.getShape(A);
		int row = shapeA[0];    
	    int col = shapeA[1]; 
		for(int i=0; i<row; i++)
		{
			for(int j=0; j<col; j++)
			{
				double value = A[i][j];
				if(value==origValue)
					A[i][j] = newValue;
				
			}
		}
		
		return A;
	}
	
	
	/**
	 * The class is necessary when the output of the activation function is in the range (-1 +1), it re-scale to range (0 , 1)
	 * @param A input matrix
	 * @return a matrix of the same dimensions as A with the values rescaled
	 */
	public static double[][] Rescaling(double[][] A)
	{
		int[] shapeA = Matrix.getShape(A);
		int row = shapeA[0];    
	    int col = shapeA[1]; 
		for(int i=0; i<row; i++)
		{
			for(int j=0; j<col; j++)
			{
				double value = A[i][j];
				value = ((value+1)/2);
				A[i][j] = value;
				
			}
		}
		
		return A;
	}
	
	
	
	public static double[][] addGaussianNoise(GaussianNoise gen, double[][] A)
	{
		
		//Calculates number of rows and columns present in first matrix    
        int[] shapeA = Matrix.getShape(A);
		int row = shapeA[0];    
        int col = shapeA[1]; 
        double noise=0;
        
        double[][] prod = new double[row][col];
        
        for(int i = 0; i < row; i++)
        {
        	for(int j = 0; j < col; j++)
            {
        		noise = gen.generate();
        		prod[i][j] = A[i][j] + noise;
//        		System.out.println("Value: "+A[i][j]+" noise: "+noise+"   new: "+prod[i][j]);
            }
        }
		
        return prod;
	}
	
	
	public static double[] straight(double[][] A)
	{
		
		//Calculates number of rows and columns present in first matrix    
        int[] shapeA = Matrix.getShape(A);
		int row = shapeA[0];    
        int col = shapeA[1]; 
        
        double[] out = new double[row*col];
        int index = 0;
        for(int i = 0; i < row; i++)
        {
        	for(int j = 0; j < col; j++)
            {
        		out[index] = A[i][j];
        		index++;
            }
        }
		
        return out;
	}
	
	
    
    
    // Concrete method to calculate minMax
    public double[] calculateMinMax(ArrayList<double[][]> elements) {
        if (elements == null || elements.isEmpty()) {
            throw new IllegalArgumentException("The input list is null or empty");
        }

        // Initialize min and max values with extreme values
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        // Loop through each matrix in the list
        for (double[][] matrix : elements) {
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[i].length; j++) {
                    // Update min and max
                    if (matrix[i][j] < min) {
                        min = matrix[i][j];
                    }
                    if (matrix[i][j] > max) {
                        max = matrix[i][j];
                    }
                }
            }
        }

        // Return the min and max as a double array
        return new double[]{min, max};
    }
	
}
