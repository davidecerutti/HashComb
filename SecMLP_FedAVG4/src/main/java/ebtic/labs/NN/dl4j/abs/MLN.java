package ebtic.labs.NN.dl4j.abs;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;

// useful links
//https://chat.chatbotapp.ai/chats/-O7NvdkPhLOiy2LFzJfb?model=gpt-3.5
//https://deeplearning4j.konduit.ai/deeplearning4j/reference/multi-layer-network
//https://chatgpt.com/share/66f1000b-37a8-800b-bc48-9ebfaae355eb
//https://github.com/deeplearning4j/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/IrisClassifier.java

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.google.common.primitives.Doubles;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.AF.ReLU;
import ebtic.labs.AF.Sigmoid;
import ebtic.labs.AF.SoftMAX;
import ebtic.labs.AF.Tanh;
import ebtic.labs.AF.ActivationFunction.Types;
import ebtic.labs.Exceptions.ConfigurationException;
import ebtic.labs.Exceptions.MatrixException;
import ebtic.labs.Federated.threads.FinalModel;
import ebtic.labs.LF.LossFunction;
import ebtic.labs.NN.EarlyStopping;
import ebtic.labs.NN.Layer;
import ebtic.labs.NN.NN;
import ebtic.labs.NN.dl4j.utils.Utils;
import ebtic.labs.metrics.Accuracy;
import ebtic.labs.metrics.History;
import ebtic.labs.utils.Matrix;
import ebtic.labs.utils.WeightsExtractor;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;


public abstract class MLN extends NN{



		private MultiLayerConfiguration mlConfiguration = null;
		protected MultiLayerNetwork model=null;
		protected Evaluation eval = null;
		protected boolean isInitialized;
		protected double gradient_clipping=1.0;
		
		
		/**
		 * Constructor for MLN
		 * @param epochs number of epochs
		 * @param learning_rate the value for the learning rate
		 * @param layers defines the number of layers(array length) and the number of neurons per layer (array values)
		 * @param activation_function activation function type
		 * @param loss_function loss function type
		 * @param ea early stopping, can be 'null' if disabled
		 * @throws ConfigurationException
		 */
		public MLN(int epochs, double learning_rate, int[] layers, ActivationFunction.Types activation_function, LossFunction.Types loss_function, EarlyStopping ea, int classes, double clipping) throws ConfigurationException {
			super(epochs, learning_rate);
			Nd4j.setDataType(DataType.DOUBLE);
			System.out.println("STAGE 2");
			this.extractor = new WeightsExtractor(epochs); 
			if(epochs<10)
			{
				System.out.println("The number of epochs should be bigger than 10!!!");
				System.exit(0);
			}
			else
			{

				this.loss_function = loss_function;
				this.ea = ea;
				this.layers = layers;
				this.classes = classes;
				this.activation_function = activation_function;
				this.gradient_clipping = clipping;
				int outputSize = this.classes;
				this.eval = new Evaluation(classes);
				if((layers==null)||(layers.length<1))
						throw new ConfigurationException("There must be at least 1 hidden layer!!!");
					// TODO Auto-generated constructor stub

				if((layers==null)||(layers.length<1))
					throw new ConfigurationException("There must be at least 1 hidden layer!!!");
				// TODO Auto-generated constructor stub

			}
			
			

		}
		
		
		protected org.nd4j.linalg.activations.Activation translateAF(ActivationFunction function)
		{
			
			switch (function.getType()) {
		    case SIGMOID   : return Activation.SIGMOID; 
		    case TANH :   return Activation.TANH; 
		    case ReLU   : return Activation.RELU; 
		    case SoftMAX : return Activation.SOFTMAX; 
		    
			}
			
			
			return null;
		}
		
		
		protected ActivationFunction.Types translateAF( org.nd4j.linalg.activations.Activation function)
		{
			
			switch (function) {
		    case SIGMOID   : return ActivationFunction.Types.SIGMOID; 
		    case TANH :   return ActivationFunction.Types.TANH; 
		    case RELU   : return ActivationFunction.Types.ReLU; 
		    case SOFTMAX : return ActivationFunction.Types.SoftMAX; 
			}
			
			
			return null;
		}
		
	
		protected void init()
		{
			if(this.mlConfiguration!=null)
			{
				// Step 2: Initialize the MultiLayerNetwork
		        this.model = new MultiLayerNetwork(this.mlConfiguration);
		        this.model.init();  
		        this.isInitialized = true;
			}
		}
		
		public MLN(int epochs, double learning_rate) {
			super(epochs, learning_rate);
			// TODO Auto-generated constructor stub
		}
		
		public MLN(int[] layers, ActivationFunction.Types af, int classes, int features, double grad) throws ConfigurationException {
			
			super(0, 0);
			this.gradient_clipping=grad;
			this.layers = layers;
			this.classes = classes;
			this.activation_function = af;
			this.eval = new Evaluation(classes);
			this.features = features;
			if((layers==null)||(layers.length<1))
					throw new ConfigurationException("There must be at least 1 hidden layer!!!");
				// TODO Auto-generated constructor stub
			int numLayers = this.layers.length+1;
			int outputSize = this.classes;
			if(outputSize==2)
				outputSize = 1;
			
			
			System.out.println("Äbout to Initialize the NETWORK!!!!");
			this.mlConfiguration = configureNetwork(numLayers, features, outputSize, layers, this.getLearningRate(), grad);
			this.init();

		}
		
		
	public abstract MultiLayerConfiguration configureNetwork(int numLayers, int inputSize, int outputSize, int[] hiddenLayerSizes, double learning_rate, double grad);
	
	
	public abstract org.nd4j.linalg.dataset.DataSet reshapeDataSet(org.nd4j.linalg.dataset.DataSet dataset);
	
	public abstract  ArrayList<Layer> initializeLayers(int input, int classes);
	
	
	
	private void printLayersSize()
	{
		for(int i = 0; i<this.Layers.size(); i++)
		{
			Layer aux = this.Layers.get(i);
    	    double[][] W = aux.get_W();
	    	int[] shape = Matrix.getShape(W);
	    	System.out.println("#Size for Layer W in input: "+shape[0]+"x"+shape[1]);
	    	   
	    	double[][] bias = aux.get_b();
	    	int[] shape2 = Matrix.getShape(bias);
	    	System.out.println("#Size for Layer b in input: "+shape2[0]+"x"+shape2[1]);
			
		}
	}
	
	
	protected abstract ArrayList<Layer> updateTransferWeights();
	protected abstract org.deeplearning4j.nn.api.Layer[] updateModelWeights();
	
	
	
	protected void aggregation()
	{
		// update the transfer layers using the aggregated weights
		super.aggregation();
		//update the transfer weights into the model weights 	
		
	}
	
	
//	protected void fit( DataSet dataset, int batchSize)
//	{
//		if(batchSize > 1)
//		{
//			List<DataSet> dataList = Arrays.asList(dataset);
//	        DataSetIterator iterator = new ListDataSetIterator<>(dataList, batchSize);
//
//	        // Step 5: Train the model
//	        this.model.fit(iterator);	
//		}
//		else
//			this.model.fit(dataset);
//		
//	}
	
	
	protected void fit( DataSet dataset, int batchSize)
	{
//		System.out.println("Data size: "+dataset.numExamples()+"       batchsize: "+batchSize);
		
		if(batchSize<dataset.numExamples())
		{
			List<DataSet> dataList = Arrays.asList(dataset);
	        DataSetIterator iterator = new ListDataSetIterator<>(dataList, batchSize);
	       
	        // Step 5: Train the model
	        this.model.fit(iterator);	
		}
		else
			this.model.fit(dataset);
		
	}
	
	 private static double evaluateModel(MultiLayerNetwork model, DataSetIterator validationIterator) {
	        // This is a placeholder for your validation score evaluation logic
	        // For instance, you could compute accuracy or loss here
		 	if(!validationIterator.hasNext())
		 		validationIterator.reset();
		 
	        return model.score(validationIterator.next()); // Just an example
	    }
	
	protected boolean validateEarlyStop(DataSetIterator validationIterator)
	{
		 boolean out = false;
		 int patience = 10; // Number of epochs to wait before stopping if no improvement
	     double bestValidationScore = Double.NEGATIVE_INFINITY; // Best score observed on validation set
	     int epochsWithoutImprovement = 0; // Counter for epochs without improvement
		
		 // Step 2: Evaluate model on validation data after each epoch
        double validationScore = evaluateModel(model, validationIterator); // Your method to evaluate the model

        // Step 3: Check if validation score improved
        if (validationScore > bestValidationScore) {
            bestValidationScore = validationScore;
            epochsWithoutImprovement = 0; // Reset counter
        } else {
            epochsWithoutImprovement++;
        }


        // Check for early stopping condition
        if (epochsWithoutImprovement >= patience)
        	out = true;
       
		return out;
	}
	
	
	@Override
	public History fit(double[][] X, double[][] labels, double batchSize) throws MatrixException, IOException {
		
		
		int[] inputShape = Matrix.getShape(X);
		int[] outputShape = Matrix.getShape(labels);
		
		int numLayers = this.layers.length+1;  // 1 input layer + 2 hidden layers + 1 output layer
		int inputSize = inputShape[1];  // e.g., for MNIST
		Matrix.straight(labels);
		
		this.features=inputSize;
		
		int[] hiddenLayerSizes = this.layers;
		int outputSize = this.classes;
		
		System.out.println("Distinct Output values: "+this.classes);
		
		if(outputSize==2)
			outputSize = 1;
		
		//initialize
		if(!this.isInitialized)
		{
			this.mlConfiguration = configureNetwork(numLayers, this.features, outputSize, layers, this.getLearningRate(), this.gradient_clipping);
			this.init();  
		}
			
		
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

		
		LossFunction lossFun = LossFunction.instanciate(this.loss_function);
		
		
//		this.printLayersSize();
		
//		this.mlConfiguration = configureNetwork(numLayers, inputSize, outputSize, hiddenLayerSizes);
//		model = new MultiLayerNetwork(this.mlConfiguration);
//		model.init();	
		org.deeplearning4j.nn.api.Layer[] modelLayers = null;
		// for random values:
	
		// for values coming from general model:
		if(!this.beInitialized)
		{
// all the previous tests where done with random initialization			
//			this.Layers = this.instanciate(Matrix.getShape(X)[1], outputSize);	
//			this.Layers = this.updateTransferWeights();
			this.Layers = initializeLayers(Matrix.getShape(X)[1], outputSize);
			this.aggregation();
			modelLayers = this.updateModelWeights();
		}
			
		else //at first iteration, the weights are initialize on the model and transferred to the transfer layers
			//initialize the model with the N4J weights
		{
			
			this.Layers = this.initializeLayers(Matrix.getShape(X)[1], outputSize);
//			this.Layers = this.instanciate(Matrix.getShape(X)[1], outputSize);	
			//initialize the model with our generated weights
//			modelLayers = this.updateModelWeights();
			this.Layers = this.updateTransferWeights();
			
			modelLayers = model.getLayers();
		}
			
			
		
		//updated weights
		
		
		
		//record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(4));
        
        
		this.extractor.setLayers(this.Layers);
		
		this.extractor.openCSV();
		
		double[][] label_t = Matrix.transpose(labels);
		try {
				if(this.classes>2)
				label_t = Utils.getMulticlass(label_t, this.classes);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				throw new IOException("There is a problem in the definition of the classifier");
			}
		DataSet trainingData = Utils.createDataSetFromArray(X, label_t);
		trainingData = this.reshapeDataSet(trainingData);
		
//		double[][] X_T = Matrix.transpose(X);
		
		// it simply displays on the System.out the advancing of the process in percetage 
		int progress = (epochs / 10);
		char[] animationChars = new char[]{'|', '/', '-', '\\'};
		int count = 0;
		//////////////////////////////////////////////////
		// loops through the epochs
		
		try 
		{
			Utils.printLayersWeight(model.getLayers(), 1, 0);
		}
		catch(Exception e) {e.printStackTrace();}
		
//		Utils.printLayersWeight(model.getLayers());
		
		
		
		 // Step 2: Create DataSetIterator from List<DataSet> for EARLY STOPPING ONLY!!!!!
		List<DataSet> dataList = Arrays.asList(trainingData);
        DataSetIterator validateIterator = new ListDataSetIterator<>(dataList, batchNum);
		
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

//			Utils.printLayersShape(modelLayers);
			
//			batchNum=-1;   // set negative to remove batching
			this.fit(trainingData, batchNum);
			
			// this code is just for early stopping, remove if no needs
			if(this.validateEarlyStop(validateIterator)) 
			{
				 System.out.println("Early stopping at epoch " + (epoch + 1) + " due to no improvement.");
		         break; // Stop training
			}
			// end!
				 
				   		        
		
		}
		Utils.printLayersWeight(model.getLayers(), 1, 0);
//		Utils.printLayersWeight(model.getLayers());
//		modelLayers = model.getLayers();
//		try {
//			Utils.transferWeights(modelLayers, this.Layers);
//		} catch (Exception e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}

		// at the endo of each epoch, updates all the Weights (after processing all the input data)
		// need to transfer all the weights back from the updated model to the object Layers
		
		this.Layers = this.updateTransferWeights();
		for(int i =0; i<this.Layers.size(); i++)
		{
			
			Layer L = Layers.get(i);
			history.setWeights(L.label, L.get_W());
			history.setBias(L.label, L.get_b());
		}

//		this.extractor.print2File();
			
			//System.out.println("Cost at epoch "+epoch+" is: "+j[0][0]);

			// if the early stopping is set
			


			//	add the output of the activation function to the history object for that epoch
			// also add the labels that have been use for the training.
			// calculate the output of the activation function on the all Labels at the end of all the batches and record it in the history
			
			INDArray output = model.output(trainingData.getFeatures());
			float[][] straight = output.toFloatMatrix();
			
			System.out.println("size1: "+straight.length+"x"+straight[0].length);
			double[][] A = Utils.convertFloatToDouble(straight);
			
			A = Matrix.transpose(A);
			
			System.out.println("size2: "+A.length+"x"+A[0].length);

			history.addTraining(A);
			history.setTrainingLabels(labels);

			
//			Accuracy accuracy = history.getAccuracy(epoch);
//			String printing = accuracy.printAccuracy(labels); 
//			System.out.println("Epoch: "+epoch+"\n"+printing);
//			System.out.println("*******************************************************************************");
//		}

		// training completed and returning the history object
		
	Evaluation eval = new Evaluation(this.classes);
	INDArray out = model.output(trainingData.getFeatures());
	
	Utils.printLayersShape(model.getLayers());
	
	try{
		eval.eval(Nd4j.create(label_t), out);
		System.out.println("Eval: "+eval.stats());
	} catch(Exception e)
	{
		e.printStackTrace();
	}
	
    
   
    
	return history;	
		
	
	}

	@Override
	public double[][] predict(double[][] X_test) throws MatrixException {
		// TODO Auto-generated method stub
		
//		this.Layers = this.instanciate(Matrix.getShape(X)[1]);
		
		Utils.printLayersShape(model.getLayers());
//		X_test = Matrix.transpose(X_test);
		
		INDArray input = Nd4j.create(X_test);
		INDArray output = model.output(input);
		float[][] straight = output.toFloatMatrix();
		double[][] A = Utils.convertFloatToDouble(straight);
		return A;
	}



	@Override
	public History fit(double[][] X, double[][] labels) throws MatrixException, IOException {
		// TODO Auto-generated method stub
		return null;
	}



	@Override
	public void instanciate(HashMap<String, double[][]> weights, HashMap<String, double[][]> bias) {
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
		
//		org.deeplearning4j.nn.api.Layer[] modelLayers = this.updateModelWeights();
		
		this.beInitialized = false;
		
	}



	@Override
	public void instanciate(HashMap<String, double[][]> weights) {
		// TODO Auto-generated method stub
		this.aggregated_weights = weights;
		
		this.beInitialized = false;
	}



	@Override
	public void instanciate(int shape, HashMap<String, double[][]> weights) {
		// TODO Auto-generated method stub
//		this.Layers = this.instanciate(shape, this.classes);
		this.Layers = this.initializeLayers(shape, this.classes);
		this.aggregated_weights = weights;
		this.aggregation();
//		this.beInitialized = false;
		this.updateModelWeights();
		
	}



	@Override
	public void instanciate(int shape, HashMap<String, double[][]> weights, HashMap<String, double[][]> bias) {
		// TODO Auto-generated method stub
//		this.Layers = this.instanciate(shape, this.classes);
		this.Layers = this.initializeLayers(shape, this.classes);
		this.aggregated_weights = weights;
		if(bias!=null)
			this.aggregated_bias = bias;
		this.aggregation();
//		this.beInitialized = false;
		this.updateModelWeights();
		
	}



	@Override
	public void save2File()  {
		// TODO Auto-generated method stub
		SimpleDateFormat formatter = new SimpleDateFormat("dd.MM.yy_HH_mm_ss");  
        Date date = new Date();
        String filename = formatter.format(date)+"_model";
        try {
			Utils.saveModel(model, filename);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public void printWeights() {
		// TODO Auto-generated method stub
		Utils.printLayersShape(this.model.getLayers());
	}

	@Override
	public void evaluate(double[][] labels, double[][] guess) {
		
		INDArray labelArray=null;
		if(guess[0].length>1)
		{
			int classes = guess[0].length;
			
			
			
			double[][] aux = Utils.getLabelsOneHot(labels[0], classes);
			labelArray = Nd4j.create(aux);
			
		}
		else 
		{
			double[][] aux = Matrix.transpose(labels);
			labelArray = Nd4j.create(aux);
		}
		
	   INDArray guessArray = Nd4j.create(guess);
	   
	   
		
	   eval.eval(labelArray, guessArray);
     
      System.out.println("Eval: "+eval.stats());
		
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
		// File where the model will be saved
        File locationToSave = new File(path+"/testModel.mod");  // Specify the file path

        // Save the model
        boolean saveUpdater = true;  // Save the optimizer's state (recommended for resuming training)
        ModelSerializer.writeModel(this.model, locationToSave, saveUpdater);

        System.out.println("Model saved to " + locationToSave.getAbsolutePath());
		
	}

}
