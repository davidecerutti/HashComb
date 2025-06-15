package ebtic.labs.NN.dl4j.tests;


import org.datavec.api.records.reader.RecordReader;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.SingletonDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import ebtic.labs.NN.dl4j.utils.Utils;

import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class FederatedTrainingCNNExample {

	
	 static int numIterations = 4;
	 static int numFeatures = 12288;
	 static int numClasses = 43;
	 static int epochs = 140;
	 static int scoreIterations = 2;
//	 static String testModel = "data/tests/N4JModels/n4j.mod";
	 static String testModel = "data/tests/N4JModels/testModel.mod";
	 static boolean training = true;
	 static boolean restore = false;
	 
	 static int batchSize = 1200;            // Batch size for training
	 static int inputHeight = 64; // Height of the image
	 static int inputWidth = 64;  // Width of the image
	 static int inputChannels = 3; // Number of channels (RGB)
	 static double learning_rate = 5e-4;
	
	public static void saveModel(String fileName, MultiLayerNetwork network) throws IOException
	{
		// File where the model will be saved
        File locationToSave = new File(fileName);  // Specify the file path

        // Save the model
        boolean saveUpdater = true;  // Save the optimizer's state (recommended for resuming training)
        ModelSerializer.writeModel(network, locationToSave, saveUpdater);

        System.out.println("Model saved to " + locationToSave.getAbsolutePath());
	}

	public static MultiLayerNetwork restoreModel(String fileName) throws IOException
	{
		File locationToLoad = new File(fileName);

        // Load the model
        MultiLayerNetwork restoredNetwork = ModelSerializer.restoreMultiLayerNetwork(locationToLoad);

        System.out.println("Model loaded from " + locationToLoad.getAbsolutePath());
        
        return restoredNetwork;

	}
	
	   private static void evaluateModel(MultiLayerNetwork network, String validationFilePath, int numFeatures, int numClasses) throws Exception {
	        // Step 1: Create a DataSetIterator for the validation dataset
	        DataSetIterator validationIterator = createDataSetIterator(new File(validationFilePath), numFeatures, numClasses);

		   
	        DataSet testData = validationIterator.next();
	        
	     // Reshape the test data from [numSamples, 12288] to [numSamples, 3, 64, 64]
	        INDArray testFeatures = testData.getFeatures();
	        testFeatures = testFeatures.reshape(testFeatures.size(0), 3, 64, 64);
	        testData.setFeatures(testFeatures);
	        // Step 2: Evaluate the model using the validation dataset
	     // Create a DataSetIterator from the DataSet
	        DataSetIterator iterator = new SingletonDataSetIterator(testData);

	        
	        double accuracy = network.evaluate(iterator).accuracy();
	        
	        // Print validation results
	        System.out.println("Validation Accuracy: " + accuracy);
	        
	        Evaluation eval = new Evaluation(numClasses);
	        INDArray output = network.output(testData.getFeatures());
	        
	        int[] labelsize = Utils.getShape(testData.getLabels());
	        System.out.println("Labels Size: "+labelsize[0]+"x"+labelsize[1]);
	        int[] outsize = Utils.getShape(output);
	        System.out.println("Guesss Size: "+outsize[0]+"x"+outsize[1]);
//	        System.out.println("LABEL content:");
//	        System.out.println(testData.getLabels());
//	        System.out.println("GUESS content:");
//	        System.out.println(output);
	        eval.eval(testData.getLabels(), output);
	        System.out.println("Eval: "+eval.stats());
	        
	        
	        // Optionally reset the validation iterator for future use
	        validationIterator.reset();
	    }
		
	
	
	
	
		public static MultiLayerConfiguration configureNetwork(int outputSize) {      
			  

			

			 MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		                .updater(new Adam.Builder().learningRate(learning_rate).build()) // Optimizer
		                .weightInit(WeightInit.XAVIER) // Weight Initialization
		                .l2(1e-4) // L2 Regularization
		                .list()
		                // First Convolutional Layer
		                .layer(0, new ConvolutionLayer.Builder(3, 3) // Filter size 3x3
		                        .nIn(3) // Input channels (RGB)
		                        .nOut(32) // Number of filters
		                        .stride(1, 1) // Stride 1x1
		                        .activation(Activation.RELU) // Activation function
		                        .build())
		                // First Max Pooling Layer
		                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // Max pooling
		                        .kernelSize(2, 2) // Pooling size 2x2
		                        .stride(2, 2) // Stride 2x2
		                        .build())
		                // Second Convolutional Layer
		                .layer(2, new ConvolutionLayer.Builder(3, 3) // Filter size 3x3
		                        .nOut(64) // Number of filters
		                        .stride(1, 1)
		                        .activation(Activation.RELU)
		                        .build())
		                // Second Max Pooling Layer
		                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
		                        .kernelSize(2, 2)
		                        .stride(2, 2)
		                        .build())
		                // Third Convolutional Layer
		                .layer(4, new ConvolutionLayer.Builder(3, 3) // Filter size 3x3
		                        .nOut(64) // Number of filters
		                        .stride(1, 1)
		                        .activation(Activation.RELU)
		                        .build())
		                // Third Max Pooling Layer
		                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
		                        .kernelSize(2, 2)
		                        .stride(2, 2)
		                        .build())
		                // Fully Connected Layer
		                .layer(6, new DenseLayer.Builder()
		                        .nOut(64) // Number of neurons
		                        .activation(Activation.RELU)
		                        .build())
		                // Dropout Layer to prevent overfitting
		                .layer(7, new DropoutLayer(0.35))
		                // Output Layer
		                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // Output layer
		                        .nOut(outputSize) // Number of output classes (e.g., 43)
		                        .activation(Activation.SOFTMAX)
		                        .build())
		                // Set Input Type to convolutional (height, width, depth)
		                .setInputType(InputType.convolutional(64, 64, 3))
		                .build();

	        return conf;
	    }
	
	
	private static void shareMergedWeightsWithLocalModels(MultiLayerNetwork mergedModel, MultiLayerNetwork[] localModels) {
	    int numLayers = mergedModel.getnLayers();

	    // Distribute the merged weights and biases back to each local model
	    for (MultiLayerNetwork localModel : localModels) {
	        for (int i = 0; i < numLayers; i++) {
	            INDArray globalWeights = mergedModel.getLayer(i).getParam("W");
	            INDArray globalBiases = mergedModel.getLayer(i).getParam("b");
	            if((globalWeights!=null)&&(globalBiases!=null))
	            {
	            	 // Set the weights and biases of the merged model into each local model
		            localModel.getLayer(i).setParam("W", globalWeights.dup());  // Use dup() to avoid sharing the same reference
		            localModel.getLayer(i).setParam("b", globalBiases.dup());
	            }
	            else
	            	System.out.println("Can't upload the weights for layer "+localModel.getLayer(i).getClass().getCanonicalName());
	           
	        }
	    }
	}

	
	
	private static MultiLayerNetwork aggregateModels(MultiLayerNetwork[] localModels) {
	    // Initialize a new global model based on the architecture of the first local model
	    MultiLayerNetwork globalModel = new MultiLayerNetwork(localModels[0].getLayerWiseConfigurations());
	    globalModel.init();

	    int numLayers = globalModel.getnLayers();
//	    String out = "";
	    // Loop through each layer to aggregate weights and biases
	    for (int i = 0; i < numLayers; i++) {
	    	
	        INDArray globalWeights = null;
	        INDArray globalBiases = null;
//	        out = out+"Layer_"+i+" [";
//	        String aux = "";
	        // Aggregate weights and biases for the i-th layer
	        int count =-1;
	        for (MultiLayerNetwork localModel : localModels) {
	        	
	        	count = count+1;
	        	Layer layer = localModel.getLayer(i);
	        	System.out.println("Processing layer type "+layer.getClass().getCanonicalName()+" Level "+i+" in Thread_"+count);
		    	if (layer instanceof org.deeplearning4j.nn.layers.convolution.ConvolutionLayer || layer instanceof org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer || layer instanceof org.deeplearning4j.nn.layers.OutputLayer) 
	            {
		    		System.out.println("Thread_"+count+" Aggregating the weights for layer "+layer.getClass().getName());
		    		 INDArray localWeights = layer.getParam("W");
//			            aux=aux+localWeights.getDouble(0,1)+", ";
			            INDArray localBiases = layer.getParam("b");

			            if (globalWeights == null) {
			                globalWeights = localWeights.dup();
			                globalBiases = localBiases.dup();
			            } else {
			            	globalWeights = globalWeights.addi(localWeights);
			            	globalBiases = globalBiases.addi(localBiases);
			            }
	            }
		    	
	        	
	          
	        }
//	        aux = aux+"]";
	        // Average the weights and biases
	        
	        if((globalWeights!=null)&&(globalBiases!=null))
	        {
	        	globalWeights = globalWeights.divi(localModels.length);
	        	globalBiases = globalBiases.divi(localModels.length);

	 	        // Set the averaged weights and biases into the global model
	 	        globalModel.getLayer(i).setParam("W", globalWeights);
	 	        globalModel.getLayer(i).setParam("b", globalBiases);
	 	        
//	 	        INDArray localWeights = globalModel.getLayer(i).getParam("W");
	        }
	        else
	        	System.out.println("Can't aggregate the weights for layer "+globalModel.getLayer(i).getClass().getCanonicalName());
	       

	        
	    }

	    return globalModel;
	}

	
	
	public static <MultiLayerNetwork, DataSet> MultiLayerNetwork[] getKeysArray(HashMap<MultiLayerNetwork, DataSet> map) {
        Set<MultiLayerNetwork> keySet = map.keySet(); // Get the keys as a Set
        return keySet.toArray((MultiLayerNetwork[]) java.lang.reflect.Array.newInstance(keySet.iterator().next().getClass(), keySet.size()));
    }
	

	public static void main(String[] args) throws Exception
	{
		
		
		String modelStorage = testModel;
		MultiLayerNetwork mergedModel= null;
		
		if(training)
		{
			mergedModel = training();
			saveModel(modelStorage, mergedModel);
		}
		
		else
		{
			mergedModel = restoreModel(modelStorage);
			evaluateModel(mergedModel, "data/HFL/validation/validation.csv" , numFeatures, numClasses);
		}
	}
	
	
    public static MultiLayerNetwork training() throws Exception {
        // Assume you have n local models and a certain number of iterations to perform
        
       
 
        MultiLayerNetwork mergedModel = null;
        HashMap<MultiLayerNetwork, DataSet> localModels = initializeLocalModels(numFeatures, numClasses, scoreIterations, restore);

        // DataSetIterators for each local model, one per client
        List<DataSetIterator> localDataIterators = getLocalDataIterators();

        // Perform federated training for a number of iterations
        for (int iteration = 0; iteration < numIterations; iteration++) {
            System.out.println("Federated Learning Iteration: " + (iteration + 1));

            // Step 1: Train each local model independently
            trainLocalModels(localModels, epochs);

            // Step 2: Extract and aggregate weights
            MultiLayerNetwork[] networks= getKeysArray(localModels);
            mergedModel = aggregateModels(networks);

            // Step 3: Share merged weights with each local model
            shareMergedWeightsWithLocalModels(mergedModel, networks);
            
            evaluateModel(mergedModel, "data/HFL/validation/validation.csv" , numFeatures, numClasses);
            
        }

        return mergedModel;
        // Final model is in localModels (the same across all)
    }

    
    private static DataSetIterator createDataSetIterator(File file, int numFeatures, int numClasses) throws Exception {
        int labelIndex = numFeatures;  // The label is in the last column (after features)
        

        int numLinesToSkip = 1;
        char delimiter = ',';
        // Use a CSVRecordReader to read the file
        CSVRecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(file));

        // Create a DataSetIterator: automatically splits features/labels from the CSV
        
        return new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
    }
    
    private static HashMap<MultiLayerNetwork, DataSet> initializeLocalModels(int numFeatures, int numClasses, int scores, boolean restore) {
    	 // Directory containing CSV files (datasets)
        
        int numLinesToSkip = 1;
    	
//    	String directoryPath = "data/HFL";  // Change this to your actual path
    	String directoryPath = "TEMP";  // Change this to your actual path
        
        // List to store the trained MultiLayerNetwork instances
    	HashMap<MultiLayerNetwork, DataSet> networks = new <MultiLayerNetwork, DataSetIterator>HashMap();
        
        // Set the number of features and number of classes for classification
       
        int outputNum = numClasses;
       
       
        int[] layers= {50,50,25,50,50};
        
      
        
        // Read all files in the directory and process each CSV file
        File directory = new File(directoryPath);
        if (directory.isDirectory()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile() && file.getName().endsWith(".csv")) {
                        try {
                            System.out.println("Training on dataset from file: " + file.getName());

                            // Step 1: Create a DataSetIterator from the CSV file
                            DataSetIterator iterator = createDataSetIterator(file, numFeatures, numClasses);
                            DataSet allData = iterator.next();
                            
                         // Reshape the test data from [numSamples, 12288] to [numSamples, 3, 64, 64]
                            INDArray trainingFeatures = allData.getFeatures();
                            trainingFeatures = trainingFeatures.reshape(trainingFeatures.size(0), 3, 64, 64);
                            allData.setFeatures(trainingFeatures);
                            
                            MultiLayerConfiguration conf = configureNetwork(outputNum);
                            
                            // Step 2: Create a new MultiLayerNetwork for this dataset
                            MultiLayerNetwork network = null;
                            if(restore)
                            	network = restoreModel(testModel);
                            else
                            	network = new MultiLayerNetwork(conf);
                         

                            // Step 3: Train the network using the dataset
                            network.setListeners(new ScoreIterationListener(scores)); // Print score every 10 iterations
                            while (iterator.hasNext()) {
                                DataSet dataSet = iterator.next();
                                network.fit(dataSet);
                            }

                            // Step 4: Add the trained network to the list
                            networks.put(network, allData);
                            
                            System.out.println("Training complete for file: " + file.getName());

                        } catch (Exception e) {
                            System.err.println("Failed to process file: " + file.getName());
                            e.printStackTrace();
                        }
                    }
                }
            }
            
            
        } else {
            System.err.println("The specified path is not a directory.");
        }
		return networks;
    }

    private static List<DataSetIterator> getLocalDataIterators() {
        // Return list of DataSetIterators for each local dataset (one per client)
        // Placeholder method
        return null;
    }

    private static void trainLocalModels(HashMap<MultiLayerNetwork, DataSet> localModels, int epochs) {
        int counter = 0;
    	Iterator iter = localModels.keySet().iterator();
    	while(iter.hasNext()) {
            MultiLayerNetwork localModel = (MultiLayerNetwork) iter.next();
            DataSet allData = localModels.get(localModel);
            System.out.println("Counter  "+counter );
            counter++;
            for(int i=0; i<epochs; i++ ) {
            	
            	localModel.fit(allData);  // Train the local model on its own data
            }
 
        }
    }
}
