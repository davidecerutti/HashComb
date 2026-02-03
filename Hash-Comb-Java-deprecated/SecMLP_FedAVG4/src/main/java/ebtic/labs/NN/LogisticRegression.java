package ebtic.labs.NN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.AF.Sigmoid;
import ebtic.labs.AF.ActivationFunction.Types;
import ebtic.labs.Exceptions.MatrixException;
import ebtic.labs.LF.LossFunction;
import ebtic.labs.metrics.Accuracy;
import ebtic.labs.metrics.History;
import ebtic.labs.utils.Matrix;

/**
 * Implements a simple Logistic Regression (1 Layer) Neural Network
 * @author maurizio
 *
 */
public class LogisticRegression extends NN{

	/**
	 * 
	 * @param epochs
	 * @param lr
	 */
	public LogisticRegression(int epochs, double lr)
	{
		super(epochs, lr);
		
	}
	
	
	@Override
	public History fit(double[][] X, double[][] labels) throws MatrixException 
	{
//		this.epochs = 100;
//		this.LR = 0.01;
		
		// this object allows to keep trace of the training
		History history = new History(1, ActivationFunction.Types.SIGMOID);
		
		
		// random initialization
		double[][] W = Matrix.random(8, 1);
		
		// static instanciation of the matrix
		//double[][] W = {{0.43623386}, {0.13099362} ,{0.92191307} ,{0.68471813} ,{0.24699183} ,{0.72607194} ,{0.65944989} ,{0.68240707}};

		
		double[][] b = Matrix.full(1, Matrix.getShape(X)[0] , 0);
		
		// Loss function is uded
		LossFunction lossFun = LossFunction.instanciate(LossFunction.Types.CE);
		
		// loop on the number of epochs
		for(int epoch = 0; epoch < epochs; epoch++)
		{
			double[][] J = null;
			
			double[][] db = Matrix.full(1, Matrix.getShape(b)[0] , Matrix.getShape(b)[1]);
			
			double[][] dW = Matrix.full(Matrix.getShape(W)[0], Matrix.getShape(W)[1], 0);
			
			System.out.println("***************************** Epoch # "+epoch+" *******************************");
			
			double[][] Z = Matrix.sum(Matrix.product(Matrix.transpose(W), Matrix.transpose(X)), b);
			
			Sigmoid sig = new Sigmoid();
			
			double[][] A = sig.exec(Z);
			
//			J = NN.lossFunction(labels, A);
			J = lossFun.exec(labels, A);
			
			J = Matrix.divide(J, labels.length);
			
			// J.shape -> 1xm (m=samples)
			
			double[][] dZ = Matrix.sub(A, labels);
			
			dW = Matrix.product(Matrix.transpose(X), Matrix.transpose(dZ));
			dW = Matrix.divide(dW, labels[0].length);
			
			db = Matrix.divide(dZ, labels[0].length);
			
			
		    //W = W -0.01 * dW
			W = Matrix.sub(W, Matrix.product(this.LR, dW));
		    //b = b - 0.01 * db
			b = Matrix.sub(b, Matrix.product(this.LR, db));
			
			history.addTraining(A);
			Accuracy accuracy = new Accuracy(A, ActivationFunction.Types.SIGMOID, 2);
			
			String printing = accuracy.toString(labels, 1000); 
			System.out.println(printing);
			System.out.println("W: "+Matrix.toString(W));
			System.out.println("*******************************************************************************");
		}
		
		
		return history;
		
	}
	
	
	/**
	 * non vectorized function, represent a single step in which the model is fit with 1 sample
	 * @param sample one single sample
	 * @param label the label of the sample
	 */
	public static void fit_1_sample(double[] sample, double label)
	{
		double J = 0;
		double db = 0;
		double z = 0;
		double b = 0;
		
		LossFunction lossFun = LossFunction.instanciate(LossFunction.Types.CE);
		
		double[] dw = new double[sample.length];
		for(int a1=0; a1<sample.length; a1++)
			dw[a1] = 0;
		
		double[] W = {0.43623386, 0.13099362 ,0.92191307 ,0.68471813 ,0.24699183 ,0.72607194 ,0.65944989 ,0.68240707};

		for(int i=0; i<sample.length; i++)
			z = z + W[i] * sample[i];
		
		z = z + b;
		
		System.out.println("z: "+z);
		Sigmoid sig = new Sigmoid();
		
		double a = sig.sigmoid(z);
		
//		double a = NN.sigmoid(z);
		
		
//		double L = NN.lossFunction(label, a);
		double L = lossFun.exec(label, a);
		
		System.out.println("Loss Function: "+L);
		
		double dz = a - label;
		
		for(int i=0; i<sample.length; i++)
			dw[i] = sample[i] * dz;
		
		db = dz;
		
		for(int i=0; i<sample.length; i++)
			W[i] = W[i] - (0.01 * dw[i]);
		
		b = b - (0.01 * db);
		
		System.out.println("Weights: "+Arrays.toString(W));
		System.out.println("bias: "+b);
	}
	
	
	/**
	 * non vectorized function, represent a single step in which the model is fit with all sample 
	 * @param samples
	 * @param labels
	 */
	public static void fit_all_samples(double[][] samples, double[][] labels)
	{
		
		double[] W = {0.43623386, 0.13099362 ,0.92191307 ,0.68471813 ,0.24699183 ,0.72607194 ,0.65944989 ,0.68240707};
		double b = 0;
		int columns = samples[0].length;
		
		LossFunction lossFun = LossFunction.instanciate(LossFunction.Types.CE);
		
		if(labels.length > 1) 
		{
			// throws some exception here!!!!!!!
		}
		
		for(int i=0; i<samples.length; i++)
		{
			
			double[] dw = new double[columns];
			for(int a1=0; a1<columns; a1++)
				dw[a1] = 0;
			
			double J = 0;
			double db = 0;
			double z = 0;
			
			

			for(int j=0; j<columns; j++)
				z = z + W[j] * samples[i][j];
			
			z = z + b;
			
			Sigmoid sig = new Sigmoid();
			
			double a = sig.sigmoid(z);
			
//			double L = NN.lossFunction(labels[0][i], a);
			double L = lossFun.exec(labels[0][i], a);
			
			System.out.println("Loss Function: "+L);
			
			double dz = a - labels[0][i];
			
			for(int j=0; j<columns; j++)
				dw[j] = samples[i][j] * dz;
			
			db = dz;
			
			for(int j=0; j<columns; j++)
				W[j] = W[j] - (0.01 * dw[j]);
			
			b = b - (0.01 * db);
			
			System.out.println("Weights: "+Arrays.toString(W));
			System.out.println("bias: "+b);
			System.out.println("-----------------------------------------------------");
			
			
			
		}
		
		
	}


	@Override
	public double[][] predict(double[][] X_test) throws MatrixException {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public void instanciate(HashMap<String, double[][]> weights, HashMap<String, double[][]> bias) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void instanciate(HashMap<String, double[][]> weights) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void instanciate(int shape, HashMap<String, double[][]> weights) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void instanciate(int shape, HashMap<String, double[][]> weights, HashMap<String, double[][]> bias) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void save2File() {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void printWeights() {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void evaluate(double[][] labels, double[][] guess) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public Types getLastAF() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public void save(String path) throws IOException {
		// TODO Auto-generated method stub
		
	}



	
}
