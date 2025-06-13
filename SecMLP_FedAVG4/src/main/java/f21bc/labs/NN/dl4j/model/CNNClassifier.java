package f21bc.labs.NN.dl4j.model;

/* *****************************************************************************
*
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
import org.datavec.api.records.reader.RecordReader;






import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.SingletonDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;


import f21bc.labs.NN.dl4j.utils.DownloaderUtility;
import f21bc.labs.NN.dl4j.utils.Utils;
import f21bc.labs.utils.Matrix;

import java.io.File;
import java.util.Arrays;

/**
* @author Adam Gibson
*/
@SuppressWarnings("DuplicatedCode")
public class CNNClassifier {

   private static Logger log = LoggerFactory.getLogger(CNNClassifier.class);

   
//	public static MultiLayerConfiguration configureNetwork(int numLayers, int inputSize, int outputSize, int[] hiddenLayerSizes, double learning_rate) {
	public static MultiLayerConfiguration configureNetwork(int outputSize) {      
		  

		int inputHeight = 64; // Height of the image
        int inputWidth = 64;  // Width of the image
        int inputChannels = 3; // Number of channels (RGB)

//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .updater(new Adam.Builder().learningRate(1e-3).build()) // Optimizer
//                .weightInit(WeightInit.XAVIER) // Weight Initialization
//                .l2(1e-3) // L2 Regularization
//                .list()
//                // First Convolutional Layer
//                .layer(0, new ConvolutionLayer.Builder(3, 3) // Filter size 3x3
//                        .nIn(3) // Input channels (RGB)
//                        .nOut(32) // Number of filters
//                        .stride(1, 1) // Stride 1x1
//                        .activation(Activation.RELU) // Activation function
//                        .build())
//                // First Max Pooling Layer
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // Max pooling
//                        .kernelSize(2, 2) // Pooling size 2x2
//                        .stride(2, 2) // Stride 2x2
//                        .build())
//                // Second Convolutional Layer
//                .layer(2, new ConvolutionLayer.Builder(3, 3) // Filter size 3x3
//                        .nOut(64) // Number of filters
//                        .stride(1, 1)
//                        .activation(Activation.RELU)
//                        .build())
//                // Second Max Pooling Layer
//                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(2, 2)
//                        .stride(2, 2)
//                        .build())
//                // Third Convolutional Layer
//                .layer(4, new ConvolutionLayer.Builder(3, 3) // Filter size 3x3
//                        .nOut(128) // Number of filters
//                        .stride(1, 1)
//                        .activation(Activation.RELU)
//                        .build())
//                // Third Max Pooling Layer
//                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(2, 2)
//                        .stride(2, 2)
//                        .build())
//                // Fully Connected Layer
//                .layer(6, new DenseLayer.Builder()
//                        .nOut(256) // Number of neurons
//                        .activation(Activation.RELU)
//                        .build())
//                // Dropout Layer to prevent overfitting
//                .layer(7, new DropoutLayer(0.5))
//                // Output Layer
//                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // Output layer
//                        .nOut(outputSize) // Number of output classes
//                        .activation(Activation.SOFTMAX)
//                        .build())
//                // Set Input Type to convolutional (height, width, depth)
//                .setInputType(InputType.convolutional(64, 64, 3))
//                .build();
        

        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam.Builder().learningRate(0.0005).build()) // Optimizer
                .gradientNormalizationThreshold(1.0) // Clip gradients above 1.0
                .weightInit(WeightInit.XAVIER) // Weight Initialization
//                .l2(1e-4) // L2 Regularization
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


        
		// Improved CNN Architecture
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .updater(new Adam.Builder().learningRate(1e-4).build()) // Slightly higher learning rate for faster convergence
//                .weightInit(WeightInit.RELU) // Weight initialization suitable for ReLU activations
//                .l2(1e-5) // L2 regularization
//                .list()
//
//                // Layer 1: Convolution -> BatchNorm -> ReLU
//                .layer(0, new ConvolutionLayer.Builder(3, 3) // 3x3 convolutional kernel
//                        .nIn(inputChannels) // Number of input channels (e.g., 3 for RGB images)
//                        .stride(1, 1) // Small stride to capture detailed features
//                        .nOut(32) // Reduced number of filters from 64 to 32 for efficiency
//                        .activation(Activation.IDENTITY) // No activation before batch norm
//                        .build())
//                .layer(1, new BatchNormalization()) // Batch Normalization after convolution
//                .layer(2, new ActivationLayer(Activation.RELU)) // ReLU Activation
//
//                // Layer 2: Max Pooling
//                .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
//                        .kernelSize(2, 2) // 2x2 max pooling
//                        .stride(2, 2)
//                        .build())
//
//                // Layer 3: Convolution -> BatchNorm -> ReLU
//                .layer(4, new ConvolutionLayer.Builder(3, 3)
//                        .stride(1, 1)
//                        .nOut(64) // Reduced filters from 128 to 64
//                        .activation(Activation.IDENTITY)
//                        .build())
//                .layer(5, new BatchNormalization()) // Batch Normalization after convolution
//                .layer(6, new ActivationLayer(Activation.RELU))
//
//                // Layer 4: Max Pooling
//                .layer(7, new SubsamplingLayer.Builder(PoolingType.MAX)
//                        .kernelSize(2, 2)
//                        .stride(2, 2)
//                        .build())
//
//                // Layer 5: Convolution -> BatchNorm -> ReLU
//                .layer(8, new ConvolutionLayer.Builder(3, 3)
//                        .stride(1, 1)
//                        .nOut(128) // Reduced filters from 256 to 128
//                        .activation(Activation.IDENTITY)
//                        .build())
//                .layer(9, new BatchNormalization()) // Batch Normalization after convolution
//                .layer(10, new ActivationLayer(Activation.RELU))
//
//                // Layer 6: Fully connected (Dense Layer)
//                .layer(11, new DenseLayer.Builder()
//                        .nOut(256) // Reduced fully connected layer from 512 to 256
//                        .activation(Activation.RELU)
//                        .build())
//
//                // Layer 7: Dropout to prevent overfitting
//                .layer(12, new DropoutLayer(0.5)) // 50% dropout
//
//                // Layer 8: Output Layer with Softmax for classification
//                .layer(13, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(outputSize) // Number of output classes
//                        .activation(Activation.SOFTMAX)
//                        .build())
//
//                .setInputType(InputType.convolutional(inputHeight, inputWidth, inputChannels)) // Specify input shape
//                .build();

        

        return conf;
    }


   
   
   public static void main(String[] args) throws  Exception {

	   // Load dataset from CSV and reshape
       RecordReader recordReader = new CSVRecordReader(1, ',');
       recordReader.initialize(new FileSplit(new File("data/tests/IJCNN.csv")));

       int labelIndex = 12288;  // The 12288th index corresponds to the labels
       int numClasses = 43;     // Number of output classes
       int batchSize = 1215;

       // Create the DataSet iterator
       DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
       DataSet allData = iterator.next();
       
//       // Reshape the input from [a, 12288] -> [a, 3, 64, 64]
//       allData.getFeatures().reshape(allData.getFeatures().size(0), 3, 64, 64);

       // Normalize the data
       DataNormalization normalizer = new NormalizerStandardize();
       normalizer.fit(allData);
       normalizer.transform(allData);

       // Split the dataset for training and testing
       SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);
       DataSet trainingData = testAndTrain.getTrain();
       DataSet testData = testAndTrain.getTest();

       
       MultiLayerConfiguration conf = configureNetwork(numClasses);
       
       
       MultiLayerNetwork network = new MultiLayerNetwork(conf);
       network.init();
       network.setListeners(new ScoreIterationListener(10));

       
       // Reshape the test data from [numSamples, 12288] to [numSamples, 3, 64, 64]
       INDArray trainingFeatures = trainingData.getFeatures();
       trainingFeatures = trainingFeatures.reshape(trainingFeatures.size(0), 3, 64, 64);
       trainingData.setFeatures(trainingFeatures);
       
       
       // Train the network with early stopping
       EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
 //          .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(2))
           .epochTerminationConditions(new ThresholdEpochTerminationCondition(0.00004))
           .scoreCalculator(new DataSetLossCalculator(new SingletonDataSetIterator(trainingData), true))
           .evaluateEveryNEpochs(5)
           .modelSaver(new LocalFileModelSaver("savedModels"))
           .build();

       EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, network, new SingletonDataSetIterator(trainingData));
       EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();


       
       
    // Assuming testData contains the flattened 2D feature array

    // Reshape the test data from [numSamples, 12288] to [numSamples, 3, 64, 64]
    INDArray testFeatures = testData.getFeatures();
    testFeatures = testFeatures.reshape(testFeatures.size(0), 3, 64, 64);
    testData.setFeatures(testFeatures);

    // Create an Evaluation object for the number of output classes
    Evaluation eval = new Evaluation(numClasses);

    // Get the network output (predictions) for the reshaped test data
    INDArray output = network.output(testData.getFeatures());

    // Evaluate the predictions against the true labels
    eval.eval(testData.getLabels(), output);

    // Print out evaluation statistics
    System.out.println(eval.stats());
       
       // Print network parameters (weights and biases)
       for (int i = 0; i < network.getLayers().length; i++) {
    	   
    	   try{
    		   
    		     INDArray weights = network.getLayer(i).getParam("W");
    	           INDArray bias = network.getLayer(i).getParam("b");
    	           try 
    	           {
    	        	   System.out.println("Layer " + i + ": W shape = " + Arrays.toString(weights.shape()) + ", b shape = " + Arrays.toString(bias.shape()));   
    	           }
    	           catch (java.lang.NullPointerException | UnsupportedOperationException e) 
    	           {
    	        	   System.out.println("Layer "+ i +" can't be print");
    	        	   };
    	           
    	           
    		   
    	   } catch(UnsupportedOperationException e)
    	   {
    		   //e.printStackTrace();
    		   System.out.println("Layer "+network.getLayer(i).getClass().getName()+" not supported");
    	   }
    	   
      
       }
   
	   
   }
}
