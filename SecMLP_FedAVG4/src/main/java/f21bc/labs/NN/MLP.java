package f21bc.labs.NN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.AF.ActivationFunction.Types;
import f21bc.labs.Exceptions.ConfigurationException;
import f21bc.labs.Exceptions.MatrixException;
import f21bc.labs.Federated.Utils;
import f21bc.labs.Federated.threads.FinalModel;
import f21bc.labs.LF.LossFunction;
import f21bc.labs.metrics.Accuracy;
import f21bc.labs.metrics.History;
import f21bc.labs.utils.Matrix;
import f21bc.labs.utils.Util;
import f21bc.labs.utils.WeightsExtractor;

/**
 * The class implements a Multi Layer Perceptron network
 * @author maurizio
 *
 */
public class MLP extends NN{

	
	
	/**
	 * Constructor for MLP
	 * @param epochs number of epochs
	 * @param learning_rate the value for the learning rate
	 * @param layers defines the number of layers(array length) and the number of neurons per layer (array values)
	 * @param activation_function activation function type
	 * @param loss_function loss function type
	 * @param ea early stopping, can be 'null' if disabled
	 * @throws ConfigurationException
	 */
	public MLP(int epochs, double learning_rate, int[] layers, ActivationFunction.Types activation_function, LossFunction.Types loss_function, EarlyStopping ea, int classes) throws ConfigurationException {
		super(epochs, learning_rate);
		System.out.println("STAGE 2");
		this.extractor = new WeightsExtractor(epochs); 
		this.classes = classes;
		if(epochs<10)
		{
			System.out.println("The number of epochs should be bigger than 10!!!");
			System.exit(0);
		}
		else
		{

			this.layers = layers;
			this.activation_function = activation_function;
			this.loss_function = loss_function;
			this.ea = ea;
			if((layers==null)||(layers.length<1))
				throw new ConfigurationException("There must be at least 1 hidden layer!!!");
			// TODO Auto-generated constructor stub
		}

	}

	
	
	public MLP(int[] layers, ActivationFunction.Types af, int classes, int features) throws ConfigurationException {
		super(0, 0);
		
		this.layers = layers;
		this.activation_function = af;
		this.classes = classes;
		this.features = features;
		if((layers==null)||(layers.length<1))
				throw new ConfigurationException("There must be at least 1 hidden layer!!!");
			// TODO Auto-generated constructor stub

	}

	
	
	
	
	/**
	 * instanciate all the Layers with previously determined weights and bias	
	 * @param input 
	 * @return the instanciated layers
	 */
	public void instanciate(HashMap<String, double[][]> weights, HashMap<String, double[][]> bias)
	{
		
		
		////////// just to print values:   CAN BE REMOVED LATER, for testing only
		Iterator iter = weights.keySet().iterator();
		while(iter.hasNext())
		{
			String layer = (String) iter.next();
			double[][] w = weights.get(layer);
			double[][] b = bias.get(layer);
			FinalModel.printWeights("W_"+layer+" Initialazing with ", w);
			FinalModel.printWeights("b_"+layer+" Initialazing with ", b);
		}
		////////////////////////////////////////////////////////////////////////
	
		this.aggregated_weights = weights;
		this.aggregated_bias = bias;
		this.beInitialized = false;
		
	}
	
	
	/**
	 * instanciate all the Layers with previously determined weights	
	 * @param input 
	 * @return the instanciated layers
	 */
	public void instanciate(int shape, HashMap<String, double[][]> weights, HashMap<String, double[][]> bias)
	{
	
		this.Layers = this.instanciate(shape, 1);
		this.aggregated_weights = weights;
		if(bias!=null)
			this.aggregated_bias = bias;
		this.aggregation();
//		this.beInitialized = false;
		
	}
	

	/**
	 * instanciate all the Layers with previously determined weights	
	 * @param input 
	 * @return the instanciated layers
	 */
	public void instanciate(HashMap<String, double[][]> weights)
	{
	
		this.aggregated_weights = weights;
		this.beInitialized = false;
		
	}
	
	
	/**
	 * instanciate all the Layers with previously determined weights	
	 * @param input 
	 * @return the instanciated layers
	 */
	public void instanciate(int shape, HashMap<String, double[][]> weights)
	{
	
		this.Layers = this.instanciate(shape, 1);
		this.aggregated_weights = weights;
		this.aggregation();
//		this.beInitialized = false;
		
	}
	

	
	
	
	

	/**
	 * Initiate forward propagation of the input through the layers
	 * @param X the input dataset
	 * @param layers the array of layers
	 * @return the output of the selected Activation Function at the end of the propagation through all the layers
	 * @throws MatrixException
	 */
	private double[][] forward(double[][] X, ArrayList<Layer> layers) throws MatrixException
	{
		double[][] A = null;
		double[][] aux = null;
		for(int i=0; i<layers.size(); i++)
		{
			Layer L = layers.get(i);
			if(i==0)
				A = L.forward(X);
			else
				A = L.forward(aux);
			aux = A;
		}

		return A;
	}

	/**
	 * Initiate the back-propagation 
	 * @param X the input dataset
	 * @param labels the labels of the input dataset
	 * @param layers the array of layers
	 * @throws MatrixException
	 */
	private void backward(double[][] X, double[][] labels, ArrayList<Layer> layers) throws MatrixException
	{

		double[][] aux = null;
		//iterate backwards through all the layers
		for(int i=(layers.size()-1); i>=0; i--)
		{
			Layer L = layers.get(i);
			//if the current layer is the last of the model (output layer)
			if(i==(layers.size()-1))
			{
				// dZ = A -Y
				double[][] A = layers.get(i-1).get_A();
				L.backward(A, labels);
			}
			// if the current layer is the first of the model
			else if(i==0)
			{
				Layer next = layers.get(i+1);
				L.backward(X, next.get_W(), next.get_dZ(), labels);
			}
			// any layer in between
			else
			{
				Layer next = layers.get(i+1);
				Layer before = layers.get(i-1);
				L.backward(before.get_A(), next.get_W(), next.get_dZ(), labels);
			}
		}


		// the code implements the following backpropagation:		
		//		L3.backward(A2, labels);
		//		
		//		L2.backward(A1, L3.get_W(), L3.get_dZ(), labels[0].length);
		//		
		//		L1.backward(X, L2.get_W(), L2.get_dZ(), labels[0].length);



	}




	@Override
	/**
	 * The method fits the Neural Network with the input matrix and the labels, also allow to define the batch side [0,1] 
	 * @param X input matrix
	 * @param labels the labels of the samples
	 * @return contains all the metrics for each epoch
	 * @throws MatrixException
	 * @throws IOException 
	 */
	public History fit(double[][] X, double[][] labels) throws MatrixException, IOException {

		
		// calculate the transpose of X
		X = Matrix.transpose(X);

		// keeps record of all the epochs in the training
		History history = new History(this.epochs, this.activation_function);


//		if(this.activation_function.equals(ActivationFunction.Types.TANH))
//		{
//			 // ONLY IF USING Tahn FUNCTION!!!!!
//	        labels = Matrix.swapValues(labels, 0, -1);
//		}
		
		LossFunction lossFun = LossFunction.instanciate(this.loss_function);

		//		this.Layers = this.instanciate(Matrix.getShape(X)[0], 0.01);

		// for random values:
		this.Layers = this.instanciate(Matrix.getShape(X)[0], 1);

		this.extractor.setLayers(Layers);
		this.extractor.openCSV();
		
		// it simply displays on the System.out the advancing of the process in percetage 
		int progress = (epochs / 100);
		char[] animationChars = new char[]{'|', '/', '-', '\\'};
		int count = 0;
		//////////////////////////////////////////////////
		// loops through the epochs
		for(int epoch = 0; epoch < epochs; epoch++)
		{
			// display the progress
			if(count==progress)
			{
				int i = (epoch / progress);
				System.out.print("Processing: " + i + "% " + animationChars[i % 4] + "\r");
				
				count = 0;
			}
			else count++;


			double[][] J = null;

			// return the output of the activation function at the end of all the forward step
			double[][] A = this.forward(X, this.Layers);


			if(this.activation_function.equals(ActivationFunction.Types.TANH))
			{
				
				A = Matrix.Rescaling(A);
			}
			
			// calculate the Loss Function
			J = lossFun.exec(labels, A);
			

			J = Matrix.divide(J, labels[0].length);

			double[][] j = Matrix.sum_by_Row(J);

//			System.out.println("Cost at epoch "+epoch+" is: "+j[0][0]);

			// if the early stopping is set
			if(this.ea!=null)
			{
				// check if the condition for stopping the iterations are reached with the current loss
				boolean stop = this.ea.stop(j[0][0]);
				if(stop)
				{
					System.out.println("2)-Loss Function converging after "+epoch+" epochs!!");
					break;
				}

			}

			// start all the backpropagation steps
			this.backward(X, labels, this.Layers);

			// at the endo of each epoch, updates all the Weights (after processing all the input data)
			for(int i =0; i<this.Layers.size(); i++)
			{
				Layer L = Layers.get(i);
				L.updateWeights();
				//print the WEIGHTS
				//System.out.println("W"+L.label+":\n"+Matrix.toString(L.get_W())+"\n");
				history.setWeights(L.label, L.get_W());
				history.setBias(L.label, L.get_b());
			}

			
			this.extractor.print2File();
			
			//			add the output of the activation function to the history object for that epoch
			System.out.println("size2: "+A.length+"x"+A[0].length);
			history.addTraining(A);
			history.setTrainingLabels(labels);

			//
			//			Accuracy accuracy = history.getAccuracy(epoch);
			//			String printing = accuracy.printAccuracy(labels); 
			//			System.out.println("Epoch: "+epoch+"\n"+printing);
			//			System.out.println("*******************************************************************************");
		}

		// training completed and returning the history object
		return history;

	}





	/**
	 * The method fits the Neural Network with the input matrix and the labels, also allow to define the batch side [0,1] 
	 * @param X input matrix
	 * @param labels the labels of the samples
	 * @param batchSize the percentage of the total input to use for batch
	 * @return contains all the metrics for each epoch
	 * @throws MatrixException
	 * @throws IOException 
	 */
	public History fit(double[][] X, double[][] labels, double batchSize) throws MatrixException, IOException {

		
		
		
		if((batchSize<=0)||(batchSize>1))
			throw new MatrixException("The batch size should be a value between 0 and 1!!");
		
		System.out.println("Input size: "+Matrix.getShape(X)[0]+"x"+Matrix.getShape(X)[1]);
		int samples_count = Matrix.getShape(X)[0];
		int batchNum = (int) (samples_count * batchSize);
		System.out.println("Batch Size: "+batchNum);
		
		int buckets = (int) (1 / batchSize);
//		System.out.println("Batch Size: "+batchNum+"    buckets: "+buckets);
		// keeps record of all the epochs in the training
		History history = new History(this.epochs, this.activation_function);

//		// Tanh returns value in range [-1 1]
//		if(this.activation_function.equals(ActivationFunction.Types.TANH))
//		{
//			 // ONLY IF USING Tahn FUNCTION!!!!!
//	        labels = Matrix.swapValues(labels, 0, -1);
//		}
		
		
		LossFunction lossFun = LossFunction.instanciate(this.loss_function);

			
		this.Layers = this.instanciate(Matrix.getShape(X)[1],1);
//		this.Layers = this.instanciate(Matrix.getShape(X)[1], 0.01);

		// for random values:
		if(!this.beInitialized)
			this.aggregation();
		
		this.extractor.setLayers(this.Layers);
		this.extractor.openCSV();
		
		double[][] X_T = Matrix.transpose(X);
		
		// it simply displays on the System.out the advancing of the process in percetage 
		int progress = (epochs / 10);
		char[] animationChars = new char[]{'|', '/', '-', '\\'};
		int count = 0;
		//////////////////////////////////////////////////
		// loops through the epochs
		
		Util.printLayersWeight(this.Layers, 1, 0);
		for(int epoch = 0; epoch < epochs; epoch++)
		{
			// display the progress
			if(count==progress)
			{
				int i = (epoch / progress);
				System.out.print("Processing: " + (i*10) + "% " + animationChars[i % 4] + "\r");
				count = 0;
			}
			else count++;

			double[][] J = null;
			double[][] j = null;
			double[][] A = null;
			int index_start = 0;
			int index_end = 0;
			
			int last_index = (batchNum*buckets) -1;
			if(last_index==samples_count)
				batchNum--;
			for(int i=0; i<buckets; i++)
			{
				
				
				index_start = (i * batchNum);
				index_end = ((i+1) * batchNum) -1;
				double[][] X_aux = Matrix.subMatrix(X, index_start, index_end);
				double[][] labels_aux = Matrix.subVector(labels, index_start, index_end);


				// calculate the transpose of X
				X_aux = Matrix.transpose(X_aux);

				// return the output of the activation function at the end of all the forward step
				
				//System.out.println("Transpose size: "+Matrix.printShape(X_aux));
				
				A = this.forward(X_aux, this.Layers);


				if(this.activation_function.equals(ActivationFunction.Types.TANH))
				{
					
					A = Matrix.Rescaling(A);
				}
				
				// calculate the Loss Function
				J = lossFun.exec(labels_aux, A);
				
				
				
				J = Matrix.divide(J, labels_aux[0].length);
				j = Matrix.sum_by_Row(J);

				
				if(j[0][0] > this.maxLoss)
					this.maxLoss = j[0][0];
				
				
				// start all the backpropagation steps
				this.backward(X_aux, labels_aux, this.Layers);

				index_start = index_end+1;
				index_end = index_end + (batchNum-1);
				
				
				
//				double counter = 0;
//				for(int h=0; h<A[0].length; h++)
//				{
//					int prediction;
//					if(A[0][h]<0.5)
//						prediction = 0;
//					else
//						prediction = 1;
//					if(prediction==labels_aux[0][h])
//						counter++;
//				}
//				
//				double accuracy = ((counter) / labels_aux[0].length);
//				System.out.println("("+counter+") / "+labels_aux[0].length+" = "+accuracy);
				
			}


			// at the endo of each epoch, updates all the Weights (after processing all the input data)
			for(int i =0; i<this.Layers.size(); i++)
			{
				double newLR = 0;
				if(this.LR<=0)
				{
					newLR = (double)(0.2/(1+(0.01*(epoch+1))));
				}
				
				Layer L = Layers.get(i);
				if(newLR>0)
					L.updateLR(newLR);
				
				L.updateWeights();
				history.setWeights(L.label, L.get_W());
				history.setBias(L.label, L.get_b());
			}

			this.extractor.print2File();
			
			//System.out.println("Cost at epoch "+epoch+" is: "+j[0][0]);

			// if the early stopping is set
			
			if(this.ea!=null)
			{				// check if the condition for stopping the iterations are reached with the current loss
				boolean stop = this.ea.stop(j[0][0]);
				if(stop)
				{
					System.out.println("1)-Loss Function converging after "+epoch+" epochs!!");
					break;
				}

			}


			//	add the output of the activation function to the history object for that epoch
			// also add the labels that have been use for the training.
			// calculate the output of the activation function on the all Labels at the end of all the batches and record it in the history
			A = this.forward(X_T, this.Layers);
//			System.out.println("size2: "+A.length+"x"+A[0].length);
			history.addTraining(A);
			history.setTrainingLabels(labels);

			
//			Accuracy accuracy = history.getAccuracy(epoch);
//			String printing = accuracy.printAccuracy(labels); 
//			System.out.println("Epoch: "+epoch+"\n"+printing);
//			System.out.println("*******************************************************************************");
		}
		Util.printLayersWeight(this.Layers, 1, 0);
		// training completed and returning the history object
		return history;

	}


	@Override
	/**
	 * The method predict the classes for a new input set
	 * @param X_test the input value to predict
]	 * @return contains all the predicted classes
	 * @throws MatrixException
	 */
	public double[][] predict(double[][] X_test) throws MatrixException
	{
//		this.Layers = this.instanciate(Matrix.getShape(X)[1]);
		X_test = Matrix.transpose(X_test);
		double[][] A = this.forward(X_test, this.Layers);

		return A;
	}



	@Override
	public void save2File() {
		// TODO Auto-generated method stub
		
	}



	@Override
	public void printWeights() {
		// TODO Auto-generated method stub
		Util.printWeights(aggregated_weights);
		Util.printWeights(aggregated_bias);
	}



	@Override
	public void evaluate(double[][] labels, double[][] guess) {
		// TODO Auto-generated method stub
		
	}



	@Override
	public Types getLastAF() {
		// TODO Auto-generated method stub
		Layer last = this.Layers.get(this.Layers.size()-1);
		return last.getAF();
	}



	@Override
	public void save(String path) throws IOException {
		// TODO Auto-generated method stub
		
	}



	


}
