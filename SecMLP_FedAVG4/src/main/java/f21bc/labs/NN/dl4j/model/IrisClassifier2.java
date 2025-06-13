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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
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

import f21bc.labs.NN.dl4j.utils.DownloaderUtility;
import f21bc.labs.NN.dl4j.utils.Utils;
import f21bc.labs.utils.Matrix;

import java.io.File;

/**
* @author Adam Gibson
*/
@SuppressWarnings("DuplicatedCode")
public class IrisClassifier2 {

   private static Logger log = LoggerFactory.getLogger(IrisClassifier2.class);

   
	public static MultiLayerConfiguration configureNetwork(int numLayers, int inputSize, int outputSize, int[] hiddenLayerSizes) {
        
        // Initialize the neural network configuration list builder
        ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
        		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        		.updater(new Adam.Builder().learningRate(1e-5).build())
 //       	    .l2(1e-5)
                .weightInit(WeightInit.RELU)
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
	
   
   
   public static void main(String[] args) throws  Exception {

       //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
       int numLinesToSkip = 1;
       char delimiter = ',';
       RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
//       recordReader.initialize(new FileSplit(new File(DownloaderUtility.IRISDATA.Download(),"iris.txt")));
       recordReader.initialize(new FileSplit(new File("data/tests/IJCNN.csv")));

       //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
//       int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
//       int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
//       int batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

       //wine
//       int labelIndex = 11;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
//       int numClasses = 9;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
//       int batchSize = 140;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)


       //spam
       int labelIndex = 12288;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
       int numClasses = 43;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
       int batchSize = 150;   //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

       
       
       DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
       DataSet allData = iterator.next();
       allData.shuffle();
       SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

       DataSet trainingData = testAndTrain.getTrain();
       DataSet testData = testAndTrain.getTest();

       //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
       DataNormalization normalizer = new NormalizerStandardize();
       normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
       normalizer.transform(trainingData);     //Apply normalization to the training data
       normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


       int outputNum = numClasses;
       

       log.info("Build model....");
      
       int[] layers= {50,50,25,50,50};
       
       MultiLayerConfiguration conf;
	
       System.out.println("Input size: "+trainingData.numInputs());
	   System.out.println("Output size: "+outputNum);
       conf = configureNetwork(layers.length+1, trainingData.numInputs(), outputNum, layers);
	
       
        
       //run the model
       MultiLayerNetwork model = new MultiLayerNetwork(conf);
       model.init();
       //record score once every 100 iterations
       model.setListeners(new ScoreIterationListener(100));

       for(int i=0; i<10000; i++ ) {
           model.fit(trainingData);
       }

       //evaluate the model on the test set
       Evaluation eval = new Evaluation(numClasses);
       INDArray output = model.output(testData.getFeatures());
       float[][] straight = output.toFloatMatrix();
       int[] labelsize = Utils.getShape(testData.getLabels());
       System.out.println("Labels Size: "+labelsize[0]+"x"+labelsize[1]);
       int[] outsize = Utils.getShape(output);
       System.out.println("Guesss Size: "+outsize[0]+"x"+outsize[1]);
//       System.out.println("LABEL content:");
//       System.out.println(testData.getLabels());
//       System.out.println("GUESS content:");
//       System.out.println(output);
       eval.eval(testData.getLabels(), output);
       log.info(eval.stats());
       System.out.println("Eval: "+eval.stats());
    // Get current parameters (weights and biases)
       INDArray currentParams = model.params();
       
       System.out.println("Number of parameters: "+currentParams.shape()[0]);
       
       Layer[] test = model.getLayers();
       for(int i = 0; i<test.length; i++)
       {
//    	   System.out.println(test[i].paramTable());
//    	   System.out.println("Layer_"+i+" size: "+test[i].params().shape()[0]);
    	   INDArray in = test[i].params();
    	   INDArray weightMatrix = test[i].getParam("W");
    	   double[][] wMatrix = weightMatrix.toDoubleMatrix();
    	   
    	   INDArray biasMatrix = test[i].getParam("b");
    	   double[] bMatrix = biasMatrix.toDoubleVector();
    	   
    	   double[] out = in.toDoubleVector();
    	   
//    	   int[] shape = Matrix.getShape(out);
//    	   System.out.println("Layer_"+i+" shape: "+shape[0]+" x "+shape[1]);
    	   System.out.println("-Layer_"+i+" size: "+out.length);
    	   System.out.println("W size "+i+" "+wMatrix.length+"x"+wMatrix[0].length);
    	   System.out.println("b size "+i+" "+bMatrix.length);
    	   System.out.println("---------");
       }
       
       
       
   }

}
