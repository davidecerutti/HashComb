package f21bc.labs.NN.dl4j.model;

import java.io.File;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.AF.ActivationFunction.Types;
import f21bc.labs.AF.SoftMAX;
import f21bc.labs.Exceptions.ConfigurationException;
import f21bc.labs.Exceptions.MatrixException;
import f21bc.labs.Federated.threads.FinalModel;
import f21bc.labs.LF.LossFunction;
import f21bc.labs.NN.EarlyStopping;
import f21bc.labs.NN.Layer;
import f21bc.labs.NN.NN;
import f21bc.labs.NN.dl4j.utils.Utils;
import f21bc.labs.metrics.Accuracy;
import f21bc.labs.metrics.History;
import f21bc.labs.utils.Matrix;
import f21bc.labs.utils.WeightsExtractor;

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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.google.common.primitives.Doubles;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;


public class dl4jMLP extends f21bc.labs.NN.dl4j.abs.MLN{


	
		public dl4jMLP(int epochs, double learning_rate, int[] layers, ActivationFunction.Types activation_function, LossFunction.Types loss_function, EarlyStopping ea, int classes, double clipping) throws ConfigurationException 
		{
			super(epochs, learning_rate, layers, activation_function,  loss_function,  ea, classes, clipping);
		}
		

		public dl4jMLP(int[] layers, ActivationFunction.Types af, int classes, int features, double clipping) throws ConfigurationException
		{
			super(layers, af, classes, features, clipping);
		}
	

		public MultiLayerConfiguration configureNetwork(int numLayers, int inputSize, int outputSize, int[] hiddenLayerSizes, double learning_rate, double grad) {
	        
			System.out.println("Initialize the neural network configuration list builder, LR: "+learning_rate);
	        // Initialize the neural network configuration list builder
	        ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
	        		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//	        		.updater(new Adam.Builder().learningRate(1e-4).build())
	        		.updater(new Adam.Builder().learningRate(learning_rate).build())
	        		.gradientNormalizationThreshold(grad) // Clip gradients above 1.0
	        	    .l2(1e-3)
//	                .weightInit(WeightInit.XAVIER)
	        	    .weightInit(WeightInit.XAVIER_UNIFORM)
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
	        	System.out.println("Last layer AF = SIGMOID");
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


		@Override
		public DataSet reshapeDataSet(DataSet dataset) {
			// TODO Auto-generated method stub
			return dataset;
		}
		

		
		protected ArrayList<Layer> updateTransferWeights()
		{
			
			org.deeplearning4j.nn.api.Layer[] modelLayers = model.getLayers();
			try 
			{
					
				this.Layers = Utils.transferWeights(modelLayers, this.Layers);
					
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return this.Layers;
		}

		
		protected org.deeplearning4j.nn.api.Layer[] updateModelWeights()
		{
			org.deeplearning4j.nn.api.Layer[] modelLayers = model.getLayers();
			try {
				modelLayers = Utils.transferWeights(this.Layers, modelLayers);
//				this.printLayersSize();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				
			}
			model.setLayers(modelLayers);
			model.getUpdater().getStateViewArray().assign(0);
			return modelLayers;
		}


		@Override
		public ArrayList<Layer> initializeLayers(int input, int classes) {
			// TODO Auto-generated method stub
			return this.instanciate(input, classes);	
		}
		

		
}
