package f21bc.labs.NN.dl4j.utils;

import java.io.File;

import java.io.IOException;
import java.util.ArrayList;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import org.deeplearning4j.nn.layers.ActivationLayer;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;

import com.google.common.primitives.Doubles;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.NN.Layer;
import f21bc.labs.utils.Matrix;



public class Utils {
	
	
	public static int[] getShape(INDArray array)
	{
		int[] out = new int[2];
		out[1] = array.columns();
		out[0] = array.rows();
		return out;
	}
	

	public static double[][] getLabelsOneHot(double[] labels, int classes)
	{
		double[][] labelsOH = new double[labels.length][classes];
		
		for(int i =0; i<labels.length; i++)
		{
			int index = (int) labels[i];
//			System.out.println("Index: "+index);
			double[] oh = new double[classes];
			for(int j=0; j<oh.length; j++)
				oh[j] = 0;
			oh[index] = 1;
			labelsOH[i] = oh;
		}
		
		return labelsOH;
		
	}
	
	
	public static void saveModel(org.deeplearning4j.nn.multilayer.MultiLayerNetwork model, String name) throws IOException
	{
		File modelFile = new File("Model/model_"+name+".mod");

        // Save the model to a file
        boolean saveUpdater = true;  // Save the state of the optimizer (updater)
        ModelSerializer.writeModel(model, modelFile, saveUpdater);

        System.out.println("Model saved successfully to " + modelFile.getAbsolutePath());
	}
	
	
	public static org.deeplearning4j.nn.multilayer.MultiLayerNetwork retrieveModel(String name) throws IOException
	{
		File modelFile = new File("Model/"+name);

        // Load the model from the file
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        System.out.println("Model loaded successfully from " + modelFile.getAbsolutePath());
		
        return network;

	}
	
	public static double[][] getMulticlass(double[][] input, int classes) throws Exception
	{
		if(input[0].length==1)
		{
			
			double[][] output = new double[input.length][classes];
			for(int i =0; i<input.length; i++)
			{
				for(int j=0; j<classes; j++)
				{
					if(input[i][0]==j)
						output[i][j]=1;
					else
						output[i][j]=0;
				}
			}
		
			return output;
		}
		
		else throw new Exception("not a vector!");
	}
	
	
	
	public static void extractShape(INDArray weights, INDArray biases) {
        // Extract the shape of the weights
        long[] weightShape = weights.shape();
        System.out.println("Shape of weights (W): " + weightShape[0] + " x " + weightShape[1]);

        // Extract the shape of the biases
        long[] biasShape = biases.shape();
        System.out.println("Shape of biases (b): " + biasShape[0]);
    }
	
	
	
    public static void printWeightShape(org.deeplearning4j.nn.api.Layer layer) {
        // Check if the layer has weights
        if (layer.numParams()>0) {
            // Get the weights parameter 'W'
            INDArray weights = layer.getParam("W");
            INDArray biasMatrix = layer.getParam("b");
            // Print the shape of the weights
            
            System.out.println("Shape of parameter 'W': " + weights.shapeInfoToString());
            System.out.println("Shape of parameter 'b': " + biasMatrix.shapeInfoToString());
            
        } else {
            System.out.println("Layer does not have parameters.");
        }
    }
	
	
	public static void printLayersShape(org.deeplearning4j.nn.api.Layer[] layers)
	{
		

	       for(int i = 0; i<layers.length; i++)
	       {
	    	   System.out.println("-Layer_"+i+":");
	    	   printWeightShape(layers[i]);
	    	   System.out.println("---------");
	    	   
	       }
		
		
	}
	
	
	public static void printLayersWeight(org.deeplearning4j.nn.api.Layer[] layers, int r, int c)
	{
		

	       for(int i = 0; i<layers.length; i++)
	       {
	    	   INDArray in = layers[i].params();
	    	   
	    	   try{
	    		   
		    	   INDArray weightMatrix = layers[i].getParam("W");
		    	   if(weightMatrix!=null)
		    	   {
			    	   long[] shape = weightMatrix.shape();
			    	   double[][] wMatrix=null;
			    	   if(shape.length==4)
			    	   {
			    		// Reshape the 4D tensor to a 2D matrix: (nOut, nIn * kernelHeight * kernelWidth)
			    	        INDArray reshapedW = weightMatrix.reshape(shape[0], shape[1] * shape[2] * shape[3]);
			    	        wMatrix = reshapedW.toDoubleMatrix();
			    	   }
			    	   
			    	   else 
			    		   wMatrix = weightMatrix.toDoubleMatrix();
			    	   System.out.println("-Layer_"+i+" W["+r+","+c+"] :"+wMatrix[r][c]);

			    	   INDArray biasMatrix = layers[i].getParam("b");
				       double[] bMatrix = biasMatrix.toDoubleVector();
			    	   try {
			    		   
			    		  
					       System.out.println("-Layer_"+i+" b["+r+","+c+"] :"+bMatrix[r]); 
			    	   }
			    	   catch(Exception e)
			    	   {
			    		   System.out.println("-Layer_"+i+" b["+0+","+0+"] :"+bMatrix[0]); 
			    	   }
			    	   
			    	   
			    	  
			    	   System.out.println("---------");
		    		   
		    	   }
		    	   
	    		   
	    		   
	    		   
	    	   } catch(UnsupportedOperationException e)
	    	   
	    	   {
//	    		   e.printStackTrace();
	    		   System.out.println("Layer "+layers[i].getClass().getName()+" not supported");
	    	   }
	    	   
	    	 

	       }
		
		
	}

	
	public static void printLayersWeight(org.deeplearning4j.nn.api.Layer[] layers)
	{
		

	       for(int i = 0; i<layers.length; i++)
	       {
	    	   String out = "";
	    	   INDArray in = layers[i].params();
	    	   if(i<layers.length-1)
	    	   {
	    		   INDArray biasMatrix = layers[i].getParam("b");
		    	   double[] bMatrix = biasMatrix.toDoubleVector();
		    	   System.out.println("-Layer_"+i+" baies:");
		    	  
		    	   for(int j = 0; j<bMatrix.length; j++)
		    		   out = out+", "+bMatrix[j];
	    	   }
	    	  
//	    	   System.out.println(out);
	       }
		
		
	}
	
	public static void printSize(HashMap<String, double[][]> allweights)
	{
		System.out.println("Empty iterator: "+allweights.keySet().isEmpty());
		Iterator iter = allweights.keySet().iterator();
		while(iter.hasNext())
		{
			String layer = (String) iter.next();
			
			double[][] w = allweights.get(layer);
			
			System.out.println("Weight size: "+w.length+"x"+w[0].length);
			
			for(int i =0; i< w.length; i++)
				for(int j =0; j< w[0].length; j++)
					System.out.println("Weight "+i+"x"+j+"  "+w[i][j]);
		}
		
	}
	
	
	// Method to convert int[] to double[]
    public static double[] intArrayToDoubleArray(int[] intArray) {
        // Create a new double[] of the same length as the int[]
        double[] doubleArray = new double[intArray.length];

        // Cast each int to double and store in the new array
        for (int i = 0; i < intArray.length; i++) {
            doubleArray[i] = (double) intArray[i];
        }

        return doubleArray;
    }
	
	
	public static double[][] convertFloatToDouble(float[][] floatArray) {
        // Get the dimensions of the input float array
        int rows = floatArray.length;
        int cols = floatArray[0].length;

        // Initialize a new double array with the same dimensions
        double[][] doubleArray = new double[rows][cols];

        // Iterate through each element of the float array
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Cast each float value to double and store it in the new double array
                doubleArray[i][j] = (double) floatArray[i][j];
            }
        }

        return doubleArray;
    }
	
    public static DataSet createDataSetFromArray(double[][] featureArray, double[][] labelArray) {
        // Convert double[][] to INDArray for features and labels
        INDArray features = Nd4j.create(featureArray);
        INDArray labels = Nd4j.create(labelArray);
        
        // Create and return the DataSet
        return new DataSet(features, labels);
    }
	
	
	public static int countDistinct(double[] values, int n)
    {
        // First sort the array so that 
        // all occurrences become consecutive
        Arrays.sort(values);
        int count = 0;
        
        // Traverse the sorted array
        for (int i = 0; i < n; i++)
        {
        	System.out.print("new value: "+values[i] +" ");
            // Move the index ahead while 
            // there are duplicates
            while (i < n - 1 && values[i] == values[i + 1])
                i++;
     
            // print last occurrence of 
            // the current element
            count = count +1;
            
        }
        
        return count;
    }
	
	// Method to extract the first column from a double[][]
    protected static double[] getFirstColumn(double[][] array) {
        double[] firstColumn = new double[array.length];  // Create an array to hold the first column

        for (int i = 0; i < array.length; i++) {
            firstColumn[i] = array[i][0];  // Extract the first element of each row
        }

        return firstColumn;
    }
	
	
	public static org.deeplearning4j.nn.api.Layer[] transferWeights(ArrayList<Layer> inputL, org.deeplearning4j.nn.api.Layer[] outputL) throws Exception
	{
		
		
		for(int i = 0; i<inputL.size(); i++)
	       {
	    	   int index = 0;
	    	   double[][] W = inputL.get(i).get_W();
	    	   double[][] b = inputL.get(i).get_b();
	    	   
//	    	   System.out.println("BIASE SHAPE: "+Matrix.printShape(b));
	    	   double[] bias = getFirstColumn(b);
	    	   
//	    	   for(int j=0; j<bias.length; j++)
//	    		   System.out.println("BIASES --> "+bias[j]);
	    	   // Convert double[][] arrays to INDArrays
	           
	    	   INDArray weights = Nd4j.create(W);
	           INDArray biases = Nd4j.create(bias);
	    	   
	           org.deeplearning4j.nn.api.Layer aux = outputL[i];
	           INDArray currentWeights = aux.getParam("W");
	           INDArray currentBiases = aux.getParam("b");
//	    	    // Create new weight INDArray (ensure it matches the shape of the current weights)
	           
	           System.out.println("Transfering weights shape:");
	           extractShape(weights, biases);
	           System.out.println("MODEL weights shape:");
	           extractShape(currentWeights, currentBiases);
	           
	           
//	           if (!weights.shape().equals(currentWeights.shape())) {
//	               throw new IllegalArgumentException("New weights shape does not match the existing weights shape.");
//	           }
//
//	           // Create new bias INDArray (ensure it matches the shape of the current biases)
//	          
//	           if (!biases.shape().equals(currentBiases.shape())) {
//	               throw new IllegalArgumentException("New biases shape does not match the existing biases shape.");
//	           }
	    	   
	    	   
//	    	   INDArray newWeights = Nd4j.create(finalA);
//	    	   aux.setParams(newWeights);
	    	  
	    	   // Set weights and biases
	    	   aux.setParam("W", weights.dup());
	    	   aux.setParam("b", biases.dup());

	    	   
//	    	   aux.clearNoiseWeightParams();
	    	   
	//    	  outputL[i] = aux;
	       }
     
     return outputL;
	}
	

	
	
	
	public static ArrayList<Layer> transferWeights(org.deeplearning4j.nn.api.Layer[] inputL, ArrayList<Layer> outputL) throws Exception
	{
		// Get current parameters (weights and biases)
        		
		if(inputL.length!=outputL.size())
			throw new Exception("Layers do not have same dimension!");
		
	       for(int i = 0; i<inputL.length; i++)
	       {
	    	   int index = 0;
	    	   
	    	// Extract the parameter table (weights and biases)
	        //   Map<String, INDArray> paramTable = inputL[i].paramTable();
	    	   
	           INDArray weightsINDArray = inputL[i].getParam("W");


	           // Convert the reshaped INDArray to a double[][] array
	           
	           double[][] weights = weightsINDArray.toDoubleMatrix();  // Convert to double[][]

	           Layer aux = outputL.get(i);
	    	   double[][] W = aux.get_W();
	    	   
	    	   int[] shape1 = Matrix.getShape(W);
	    	   int[] shape2 = Matrix.getShape(weights);
	    	   System.out.println(i+"_Size for N4J Layer W in input: "+shape2[0]+"x"+shape2[1]);
	    	   System.out.println(i+"_Size for My Layer W in input: "+shape1[0]+"x"+shape1[1]);
//	    	   for(int a=0; a<shape2[0]; a++)
//	    	   {
//	    		   for(int b=0; b<shape2[1]; b++)
//	    		   {
//	    			W[a][b] = weights[a][b];
//	    		   }
//	    		   
//	    	   }
	    	   
	    	   
	    	   
	    	// Extract biases (b)
	           INDArray biasINDArray =  inputL[i].getParam("b");
	           
	           double[] biases = biasINDArray.toDoubleVector();  // Convert to double[][]
	           double[][] bi = aux.get_b();
	           bi = new double[biases.length][1];
	           shape1 = Matrix.getShape(bi);
//	    	   shape2 = Matrix.getShape(biases);
	           System.out.println(i+"_Size for N4J Layer b in input:"+biases.length+"  value:"+biases[0]);
	           System.out.println(i+"_Size for My Layer b in input: "+shape1[0]+"x"+shape1[1]);
	    	   for(int a=0; a<biases.length; a++)
	    	   {
	    		   
	    			   bi[a][0] = biases[a];		   
	    	   }
	    	 
	    	     
//	
	    	   aux.initialize(weights, bi);
//	    	   outputL.remove(i);
//	    	   outputL.add(i, aux);
	       }
        
        return outputL;
	}
	

	
	protected static ActivationFunction.Types translateAF( org.nd4j.linalg.activations.Activation function)
	{
		
		switch (function) {
	    case SIGMOID   : return ActivationFunction.Types.SIGMOID; 
	    case TANH :   return ActivationFunction.Types.TANH; 
	    case RELU   : return ActivationFunction.Types.ReLU; 
	    case SOFTMAX : return ActivationFunction.Types.SoftMAX; 
		}
		
		
		return null;
	}
	
	
    public static Activation convertIActivationToND4J(IActivation activation) {
        // Get the activation function name
        String activationName = activation.getClass().getSimpleName();

        // Match the activation name to the corresponding ND4J Activation
        switch (activationName) {
            case "ActivationSigmoid":
                return Activation.SIGMOID;
            case "ActivationTANH":
                return Activation.TANH;
            case "ActivationReLU":
                return Activation.RELU;
            case "ActivationSoftmax":
                return Activation.SOFTMAX;
            case "ActivationLeakyReLU":
                return Activation.LEAKYRELU;
            // Add other cases as necessary
            default:
//                throw new IllegalArgumentException("Unknown activation function: " + activationName);
            	 return Activation.RELU;
        }
    }
	
	private static org.nd4j.linalg.activations.Activation getActivationFunction(org.deeplearning4j.nn.api.Layer layer) {
        
		IActivation iact = null;

		if (layer instanceof DenseLayer) {
			iact = ((DenseLayer) layer).layerConf().getActivationFn(); // For Dense layers
        } else if (layer instanceof ConvolutionLayer) {
        	iact = ((ConvolutionLayer) layer).layerConf().getActivationFn(); // For Convolution layers
        } else if (layer instanceof OutputLayer) {
        	iact = ((OutputLayer) layer).layerConf().getActivationFn(); // For Output layers
        } else if (layer instanceof ActivationLayer) {
        	iact = ((ActivationLayer) layer).layerConf().getActivationFn(); // For Activation layers
        }

        	
		if(iact!=null)
		{
			return convertIActivationToND4J(iact);
		}
        // Add other layer types as needed

		else
		{
			
			return null; // Return null if no activation function is applicable
		}
			
    }
	
	

	
	   public static ArrayList<Layer> transferCNNWeights(org.deeplearning4j.nn.api.Layer[] inputL) {
	        // Create an ArrayList to hold the weights
	        ArrayList<Layer> weightsList = new ArrayList<>();
	        int count =-1;
	        // Iterate through each layer in the input array
	        for (org.deeplearning4j.nn.api.Layer layer : inputL) {
	        	count=count+1;
	            try 
	            {  
	        		System.out.println("The Layer has instance of "+layer.getClass().getName());
	                // Try to get the weight parameters 'W'
	                INDArray weightsParam = layer.getParam("W"); // Get the weights
	                
	             // Get the shape of the weights, should be [nOut, nIn, kernelHeight, kernelWidth]
	                long[] shape = weightsParam.shape(); // This should return [32, 3, 5, 5]
	                
	                // Ensure it's a 4D tensor
	                if ((shape.length != 4)&&(shape.length != 2)) {
	                    throw new IllegalArgumentException("Expected shape [nOut, nIn, kernelHeight, kernelWidth], but got: " + java.util.Arrays.toString(shape));
	                }
	                
	                double[][] weights = null;
	                
	                if(shape.length==4)
	                {
	                    // Reshape the 4D tensor to a 2D matrix: (nOut, nIn * kernelHeight * kernelWidth)
		                INDArray reshapedW = weightsParam.reshape(shape[0], shape[1] * shape[2] * shape[3]);
		                
		                weights = reshapedW.toDoubleMatrix(); // Ensure weights are in the correct format
		     
	                }
	                else
	                {
	                	weights = weightsParam.toDoubleMatrix();
	                }
	                
	                int[] shapeW = Matrix.getShape(weights);
	                
	 	            INDArray biasINDArray =  layer.getParam("b");
	 	            double[] biases = biasINDArray.toDoubleVector();  // Convert to double[][]
	 	            double[][] bi = new double[biases.length][1];
	 	            
	 	           for(int a=0; a<biases.length; a++)
		    	   {
		    		   
		    			   bi[a][0] = biases[a];		   
		    	   }
	 	            
	 	            ActivationFunction.Types af = translateAF(getActivationFunction(layer));
	 	            System.out.println("Activation Function: "+af);
	 	            Layer newLayer = new Layer("Layer_"+count, 0.001, af);
	 	            newLayer.initialize(weights, bi);
	 	            weightsList.add(newLayer);
	 	            System.out.println("_Size for N4J Layer W in input: "+weightsParam.shapeInfoToString());
		    	    System.out.println("_Size for My Layer W in input: "+shapeW[0]+"x"+shapeW[1]);
//	                weightsList.add(weights); // Add to the list
	            } catch (IllegalArgumentException| NullPointerException | UnsupportedOperationException e){
	                // Handle the case where the layer does not have weights
	                // You can print a debug message if needed
	            	//e.printStackTrace();
	                System.out.println("Layer does not have weights: " + layer.getClass().getSimpleName());
	            }
	        }

	        return weightsList; // Return the list of weights
	    }
	
	
	
    public static org.deeplearning4j.nn.api.Layer[] transferCNNWeights(ArrayList<Layer> inputL, org.deeplearning4j.nn.api.Layer[] outputL) {
        // Ensure that the input list size matches the output layers size
//        if (inputL.size() != outputL.length) {
//            throw new IllegalArgumentException("Input weights size does not match output layers size.");
//        }

        // Iterate through each layer's weights
    	int index = 0;
        for (int i = 0; i < outputL.length; i++) {
        	
        	
        	// Check if the current layer has weights
            if (outputL[i] instanceof ConvolutionLayer || outputL[i] instanceof DenseLayer || outputL[i] instanceof OutputLayer || outputL[i] instanceof ActivationLayer) 
            {
            	double[][] W = inputL.get(index).get_W();
    	    	double[][] b = inputL.get(index).get_b();
                // Set the weights parameter for the layers that have weights
                outputL[i].setParam("W", Nd4j.create(W));
                outputL[i].setParam("b", Nd4j.create(b));
                index = index+1;
            }
        	
	    	
            
            
            // Subsampling layers do not have weights, so we skip them
        }

        double[] mM = f21bc.labs.utils.Util.calculateMinMax(inputL);
        System.out.println("Min weight: "+mM[0]+"      MAX weight: "+mM[1]);
        return outputL; // Return the updated array of layers
    }
	
		

	
	
	

	
	
}
