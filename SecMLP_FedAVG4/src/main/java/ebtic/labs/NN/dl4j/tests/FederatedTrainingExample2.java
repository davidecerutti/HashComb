package ebtic.labs.NN.dl4j.tests;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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

public class FederatedTrainingExample2 {

	
	 static int numIterations = 10;
	 static int numFeatures = 12288;
	 static int numClasses = 43;
	 static int epochs = 1000;
	 static int scoreIterations = 10;
//	 static String testModel = "data/tests/N4JModels/n4j.mod";
	 static String testModel = "data/tests/N4JModels/testModel.mod";
	 static boolean training = true;
	 static boolean restore = false;
	 static int batchSize = 1215;            // Batch size for training
	
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
	        // Step 2: Evaluate the model using the validation dataset
	        double accuracy = network.evaluate(validationIterator).accuracy();
	        
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
		
	
	
	
	
public static MultiLayerConfiguration configureNetwork(int numLayers, int inputSize, int outputSize, int[] hiddenLayerSizes) {
        
        // Initialize the neural network configuration list builder
        ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
        		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        		.updater(new Adam.Builder().learningRate(1e-4).build())
        	    .l2(1e-5)
                .weightInit(WeightInit.XAVIER_UNIFORM)
 //       		.weightInit(new NormalDistribution(0, 0.03)) 
                .activation(Activation.RELU)
                .list();
        
        // Input layer
        listBuilder.layer(0, new DenseLayer.Builder()
                .nIn(inputSize)
                .nOut(hiddenLayerSizes[0])
                .build());
        
        // Hidden layers (dynamically based on numLayers)
        for (int i = 1; i < numLayers - 1; i++) {
            listBuilder.layer(i, new DenseLayer.Builder()
                    .nIn(hiddenLayerSizes[i - 1])
                    .nOut(hiddenLayerSizes[i])
                    .build());
        }
        
        
        
        if(outputSize>1)
        {
        	System.out.println("Last layer AF = SOFTMAX");
	        // Output layer in case of multi-class cross entropy loss function
	        listBuilder.layer(numLayers - 1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	                .activation(Activation.SOFTMAX)
	                .nIn(hiddenLayerSizes[numLayers - 2])
	                .nOut(outputSize)
	                .build());
        }
        else
        {
        	System.out.println("Last layer AF = Sigmoid");
   	     // Output layer in case of binary multi-label classification
	        listBuilder.layer(numLayers - 1, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
	                .activation(Activation.SIGMOID)
	                .nIn(hiddenLayerSizes[numLayers - 2])
	                .nOut(outputSize)
	                .build());

        }
        // Build and return the configuration
        return listBuilder.build();
    }
	
	
	private static void shareMergedWeightsWithLocalModels(MultiLayerNetwork mergedModel, MultiLayerNetwork[] localModels) {
	    int numLayers = mergedModel.getnLayers();

	    // Distribute the merged weights and biases back to each local model
	    for (MultiLayerNetwork localModel : localModels) {
	        for (int i = 0; i < numLayers; i++) {
	            INDArray globalWeights = mergedModel.getLayer(i).getParam("W");
	            INDArray globalBiases = mergedModel.getLayer(i).getParam("b");

	            // Set the weights and biases of the merged model into each local model
	            localModel.getLayer(i).setParam("W", globalWeights.dup());  // Use dup() to avoid sharing the same reference
	            localModel.getLayer(i).setParam("b", globalBiases.dup());
	        }
	    }
	}

	
	
	private static MultiLayerNetwork aggregateModels(MultiLayerNetwork[] localModels) {
	    // Initialize a new global model based on the architecture of the first local model
	    MultiLayerNetwork globalModel = new MultiLayerNetwork(localModels[0].getLayerWiseConfigurations());
	    globalModel.init();

	    int numLayers = globalModel.getnLayers();
	    String out = "";
	    // Loop through each layer to aggregate weights and biases
	    for (int i = 0; i < numLayers; i++) {
	        INDArray globalWeights = null;
	        INDArray globalBiases = null;
	        out = out+"Layer_"+i+" [";
	        String aux = "";
	        // Aggregate weights and biases for the i-th layer
	        for (MultiLayerNetwork localModel : localModels) {
	            INDArray localWeights = localModel.getLayer(i).getParam("W");
	            aux=aux+localWeights.getDouble(0,1)+", ";
	            INDArray localBiases = localModel.getLayer(i).getParam("b");

	            if (globalWeights == null) {
	                globalWeights = localWeights.dup();
	                globalBiases = localBiases.dup();
	            } else {
	                globalWeights.addi(localWeights);
	                globalBiases.addi(localBiases);
	            }
	        }
	        aux = aux+"]";
	        // Average the weights and biases
	        globalWeights.divi(localModels.length);
	        globalBiases.divi(localModels.length);

	        // Set the averaged weights and biases into the global model
	        globalModel.getLayer(i).setParam("W", globalWeights);
	        globalModel.getLayer(i).setParam("b", globalBiases);
	        
	        INDArray localWeights = globalModel.getLayer(i).getParam("W");
	        System.out.println(aux);
	        System.out.println("AVG: "+localWeights.getDouble(0,1)+"\n");
	        
	    }

	    return globalModel;
	}

	
	
	public static <MultiLayerNetwork, DataSet> MultiLayerNetwork[] getKeysArray(HashMap<MultiLayerNetwork, DataSet> map) {
        Set<MultiLayerNetwork> keySet = map.keySet(); // Get the keys as a Set
        return keySet.toArray((MultiLayerNetwork[]) java.lang.reflect.Array.newInstance(keySet.iterator().next().getClass(), keySet.size()));
    }
	

	public static void main(String[] args) throws Exception
	{
		
		
		String modelStorage = testModel;;
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
    	
    	String directoryPath = "data/HFL";  // Change this to your actual path
        
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
                            
                            MultiLayerConfiguration conf = configureNetwork(layers.length+1, numFeatures, outputNum, layers);
                            
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
