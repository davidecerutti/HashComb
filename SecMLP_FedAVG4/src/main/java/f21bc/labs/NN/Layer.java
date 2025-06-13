package f21bc.labs.NN;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.Exceptions.MatrixException;
import f21bc.labs.utils.Matrix;
/**
 * Implements a layer in the Network
 * @author maurizio
 *
 */
public class Layer {

	// definition of all the Matrix used in each layer
	private double[][] W;
	private double[][] b;
	private double[][] A;
	private double[][] Z;
	private double[][] dZ;
	private double[][] dW;
	private double[][] db;
	private double lR;
	
	private ActivationFunction af;
	//public int num;
	public String label;
	
	/**
	 * Constructor for the Layer
	 * @param label just a name for the layer
	 * @param lR learning rate value
	 * @param type Activation Function type
	 */
	public Layer(String label, double lR, ActivationFunction.Types type)
	{
		this.lR = lR;
		this.label = label;
		
		this.af = ActivationFunction.instanciate(type);
		
	}
	
	
	public ActivationFunction.Types getAF()
	{
		return this.af.getType();
	}
	
	/**
	 * initialize the Weights and bias with random values providing the shape
	 * @param row 
	 * @param column
	 */
	public void initialize(int row, int column)
	{
		W=Matrix.random(row, column);
		b=Matrix.full(column, 1, 0); 
	}

	/**
	 * Initialize with already defined Weights and Bias 
	 * @param W
	 * @param b
	 */
	public void initialize(double[][] W, double[][] b)
	{

		this.W=W;
		this.b=b;
	}

	
	
	/**
	 * Initialize with already defined Weights 
	 * @param W
	 * @param b
	 */
	public void initialize(double[][] W)
	{
		this.W = W;
//		this.b=b;
		this.b=Matrix.full(this.W[0].length, 1, 0); 
		
	}
	
	/**
	 * Return the size of the Weights matrix
	 * @return array of integer (2), containing the shape of W
	 * @throws Exception
	 */
	public int[] getSize() throws Exception
	{
		
			int[] out = {this.W.length, this.W[0].length};
			return out;
		
			
	}
	
	/**
	 * Implement the forward step for the layer
	 * @param X input matrix
	 * @return the output matrix
	 * @throws MatrixException
	 */
	public double[][] forward(double[][] X) throws MatrixException
	{
		// product of W transpose and input X
		double[][] Z = Matrix.product(Matrix.transpose(W), X);
		
		// sum all the bias vector
		Z = Matrix.sum(Z, b);
		
//		calculate the activation function
		this.A = this.af.exec(Z);
				
		return this.A;		
	}
	
	
	/**
	 * Implement the calculation of dW and db in the backpropagation step for the layer
	 * @param A the input matrix 
	 * @param labels the labels for the input
	 * @throws MatrixException
	 */
	private void backward(double[][] A, int labels) throws MatrixException
	{
		// calculate at the step x, the value -->  dW[x] = (dZ[x] * TA[x-1])/m
		this.dW = Matrix.product(A, Matrix.transpose(this.dZ)); 
		this.dW = Matrix.divide(dW, labels);
		
//		System.out.println("DW"+this.label+" "+Matrix.toString(dW));
		
		
		//double[][] db = Matrix.divide(this.dZ, labels);
		
		// calculate at the step x --> db = dZ
		
		this.db = Matrix.sum_by_Col(this.dZ);
		this.db = Matrix.divide(db, labels);

		// Weights update has been moved outside, these lines are not use anymore!
//		W = Matrix.sub(W, Matrix.product(this.lR, dW));
//		b = Matrix.sub(b, Matrix.product(this.lR, db));
	}
	
	
	/**
	 * Implements the calculation of dZ in the backpropagation step in the inner layer X
	 * @param A is the activation function result from layer X-1
	 * @param w Weights at layer X+1
	 * @param dz derivative of Z at layer X+1
	 * @param labels labels of the input
	 * @throws MatrixException
	 */
	public void backward(double[][] A, double[][] w, double[][] dz, double[][] labels) throws MatrixException
	{
		
//		this.dZ = Matrix.hadamard(Matrix.product(w, dz), NN.sigmoid_derivative(this.A));
		
		// calculate at the step X --> dZ[x] = WdZ[x+1] * AF_derivative(A[x-1]) 
		this.dZ = Matrix.hadamard(Matrix.product(w, dz), this.af.exec_derivative(this.A));
		
		this.backward(A, labels[0].length);		
	}

	/**
	 * Implements the calculation of dZ in the backpropagation step in the Output layer
	 * @param A last activation function results
	 * @param y the labels of the input
	 * @throws MatrixException
	 */
	public void backward(double[][] A, double[][] y) throws MatrixException
	{
		
		this.dZ = Matrix.sub(this.A, y);
		
		this.backward(A, y[0].length);
	}
	
	/**
	 * Implements the final step of weight update
	 * @throws MatrixException
	 */
	public void updateWeights() throws MatrixException
	{
//		System.out.println("W"+label+": "+Matrix.getShape(W)[0]+" "+Matrix.getShape(W)[1]);
//		System.out.println("b"+label+" "+Matrix.getShape(b)[0]+" "+Matrix.getShape(b)[1]);
		
		W = Matrix.sub(W, Matrix.product(this.lR, this.dW));
		b = Matrix.sub(b, Matrix.product(this.lR, this.db));

//test		
//		int[] b_shape = Matrix.getShape(b);
//		b = Matrix.full(b_shape[0], 1, 0);
		
		
		//  ---- VERY IMPORTANT !!!!! ----------------------------------------------------
		// at the end of each iteration the matrix of bias is D*n where D is the neurons at the layer and n is the number of sample,
		// this creates a problem when the model is used for prediction (in general with a set of different shape than the one used for training)
		// as it generates an error in the calculation of Z when trying to SUM matrix of different shapes.
		// for this reason I bring back the shape of db to Dx1 by calculating the average value across the column
		b = Matrix.average_by_row(b);
	}
	/**
	 * 
	 * @return the mAtrix of Weights
	 */
	public double[][] get_W()
	{
		return this.W;
	}
	
	/**
	 * 
	 * @return the bias
	 */
	public double[][] get_b()
	{
		return this.b;
	}
	
	/**
	 * 
	 * @return the output of the activation function
	 */
	public double[][] get_A()
	{
		return this.A;
	}
	
	/**
	 * 
	 * @return the sum of weights and bias
	 */
	public double[][] get_Z()
	{
		return this.Z;
	}
	
	/**
	 * 
	 * @return the derivative of Z
	 */
	public double[][] get_dZ()
	{
		return this.dZ;
	}
	
	
	public void updateLR(double lr)
	{
		this.lR=lr;
	}
}
