package f21bc.labs.NN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.Exceptions.MatrixException;
import f21bc.labs.LF.LossFunction;
import f21bc.labs.metrics.History;
import f21bc.labs.utils.Matrix;
import f21bc.labs.utils.WeightsExtractor;

/**
 * abstract class defining a Neural Network model
 * @author maurizio
 *
 */
public abstract class NN {
	
	protected int epochs;
	protected double LR;
	protected double batchSize = 0.25;
	
	//defines the number of layers(array length) and the number of neurons per layer (array values)
	protected int[] layers;

	//activation function type
	protected ActivationFunction.Types activation_function;

	//loss function type
	protected LossFunction.Types loss_function;

	// if the Early stopping is defined
	protected EarlyStopping ea = null;

	// contain the list of Hidden Layers objects
	protected ArrayList<Layer> Layers=null;
	
	protected WeightsExtractor extractor = null;
	
	protected boolean beInitialized = true;
	
	protected HashMap<String, double[][]> aggregated_weights=null;
	protected HashMap<String, double[][]> aggregated_bias=null;
	
	protected double maxLoss=1;
	
	protected int classes = 2;
	
	protected int features;

	
	
	/**
	 * Constructor for the model
	 * @param epochs number of epochs
	 * @param learning_rate value of learning rate, usually 0.001
	 */
	public NN(int epochs, double learning_rate)
	{
		
		this.epochs = epochs;
		this.LR = learning_rate;
	}
	
	/**
	 * 
	 * @param X samples in input rowsXcolumns
	 * @param labels all the labels of the training set in the form rowsX1
	 * @return contains the results of the training
	 * @throws MatrixException 
	 * @throws IOException 
	 */
	public abstract History fit(double[][] X, double[][] labels) throws MatrixException, IOException;
	
	/**
	 * 
	 * @param X_test samples for prediction 
	 * @return array of results in form rowsx1
	 * @throws MatrixException
	 */
	public abstract double[][] predict(double[][] X_test) throws MatrixException;
	
	
	public abstract void instanciate(HashMap<String, double[][]> weights, HashMap<String, double[][]> bias);
	
	
	public abstract void instanciate(HashMap<String, double[][]> weights);
	
	
	public abstract void instanciate(int shape, HashMap<String, double[][]> weights);
	
	public abstract void instanciate(int shape, HashMap<String, double[][]> weights, HashMap<String, double[][]> bias);
	
	
	public abstract void save2File();
	
	
	public abstract void printWeights();
	
	
	public abstract void evaluate(double[][] labels, double[][] guess);
	
	public abstract ActivationFunction.Types getLastAF();
	
	public abstract void save(String path) throws IOException;
	
	
	
	
	public void setClasses(int value)
	{
		this.classes = value;
	}
	
	
	public int getEpochs()
	{
		return this.epochs;
	}
	
	
	public double getLearningRate()
	{
		return this.LR;
	}

	
	public WeightsExtractor getWeightExtractor() {
		return this.extractor;
	}

	
	public void fit(double[][] X, int[] labels) {
		// TODO Auto-generated method stub
		
	}

	public History fit(double[][] X, double[][] labels, double batchSize) throws MatrixException, IOException {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	public double getBatchSize()
	{
		return this.batchSize;
		}
	
	public void setBatchSize(double batchSize) throws Exception
	{
		if((batchSize<=0)||(batchSize>1))
			throw new Exception("The batch size should be a value between 0 and 1!!");
		else 
			this.batchSize = batchSize;
	}
	
	
	/**
	 * instanciate all the Layers with random values	
	 * @param input 
	 * @return the instanciated layers
	 */
	protected ArrayList<Layer> instanciate(int input, int classes)
	{
		// set random values
		//return this.instanciate(input, 0);
		
		//set same value for weights
		return this.instanciate(input, 1, classes);
	}

	/**
	 * instanciate all the layers with default value. In general, is used for testing. Make the system deterministic
	 * @param input the value indicates the number of features of the samples used for training.
	 * @param init_value the default value, all the weights will have same value.
	 * @return the instanciated layers
	 */
	protected ArrayList<Layer> instanciate(int input, double init_value, int classes)
	{
		// the init_value!=0 is only used to avoid the random creation... for testing purpose only, to monitor all the steps.
		ArrayList<Layer> output = new ArrayList<Layer>();
		int last_layer=0;
		int neurons=0;

		if(classes==0 || classes==2)
			classes = 1;
		
		// the random values are all in range [-5, 5]. I've tried to randomize in range [0, 1] but it converges much slower (or doesn't converge at all)
		double min = -0.5;
		double max = 0.5;
		// the array layers define the number of hidden layers and neurons per layer
		for(int i=0; i<layers.length; i++)
		{
			neurons = layers[i];
			int aux;
			//the shape of Weights in the first layer is indicated by the number of samples in dataset and the neurons for that layer  
			if(i==0)
				aux=input;
			else  // in all the other cases the shape is given by the output of the previous layer and the neurons for that layer 
				aux = last_layer;

			double[][] W = null;
			if( init_value == 0)
			{
				//				System.out.println("WEIGHTS initialized randomly between [0,1]");				
				//				W = Matrix.random(aux, neurons);
				//				System.out.println("WEIGHTS initialized randomly between [-5,5]");	
				
				
				//this code implements the Xavier method for random initialization
				//min= -(1/Math.sqrt(neurons));
				//max= +(1/Math.sqrt(neurons));
				
				//W = Matrix.random(aux, neurons, min, max);
				W = Matrix.MSRA(aux, neurons);
			}

			else // just for testing
				//W = Matrix.full(aux, neurons, init_value);
				W = Matrix.full(aux, neurons, min, max);

			//			System.out.println("W"+i+" "+Matrix.getShape(W)[0]+" "+Matrix.getShape(W)[1]);

			double[][] b = Matrix.full(neurons, 1 , 0);

			//initialize the object layer with the W and bias, loss function and activation function 
			Layer L = new Layer(String.valueOf(i), this.LR, this.activation_function);
			L.initialize(W, b);

			output.add(i, L);
			last_layer = neurons;

		}

		// this last part is just for the OUTPUT layer which requires some changes as the shape is always Nx1
		double[][] Wo;
		if( init_value == 0)
		{
			//			Wo = Matrix.random(last_layer, 1);
			Wo = Matrix.MSRA(last_layer, classes);
		}

		else 
			//Wo = Matrix.full(last_layer, 1, init_value);
			Wo = Matrix.full(last_layer, classes, min, max);

		//		System.out.println("W0 : "+Matrix.getShape(Wo)[0]+" "+Matrix.getShape(Wo)[1]);
		double[][] bo = Matrix.full(classes, 1 , 0);

		//if the activation function is Relu, the last layer AF is set to sigmoid as it is difficult to interpret results within range [0, R]
		ActivationFunction.Types last_af = this.activation_function;
		if(this.activation_function.equals(ActivationFunction.Types.ReLU))
			last_af = ActivationFunction.Types.SIGMOID;
		if(classes >1)
			last_af = ActivationFunction.Types.SoftMAX;
		
		Layer L = new Layer(String.valueOf(layers.length), this.LR, last_af);
		L.initialize(Wo, bo);

		output.add(layers.length, L);
	
		
		return output;
	}


	/**
	 * instanciate all the Layers with previously determined weights	
	 * @param input 
	 * @return the instanciated layers
	 */
	protected void aggregation()
	{
				
		Iterator<Layer> layerIter = this.Layers.iterator();
		int index = 0;
		while(layerIter.hasNext())
		{
			
			Layer layer =  layerIter.next();
			String label = layer.label;
			System.out.println("Init Layer_"+label);
			
			double[][] neWeights = this.aggregated_weights.get(label);
			if(this.aggregated_bias!=null)
			{
				double[][] neBias = this.aggregated_bias.get(label);
				layer.initialize(neWeights, neBias);
			}
			else	
				layer.initialize(neWeights);
			//this.Layers.add(index, layer);
			index++;
		}
		
		//return this.Layers;
	}

	
}

