package ebtic.labs.Federated.threads;

import java.io.File;


import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import org.apache.hadoop.mapred.FileSplit;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.Exceptions.ConfigurationException;
import ebtic.labs.Exceptions.MatrixException;
import ebtic.labs.Federated.ModelAggregator;
import ebtic.labs.Federated.Utils;
import ebtic.labs.Federated.hashComb.HCDecoder;
import ebtic.labs.Federated.hashComb.HCEncoder;
import ebtic.labs.Federated.utils.EncodedWeight;
import ebtic.labs.LF.LossFunction;
import ebtic.labs.NN.EarlyStopping;
import ebtic.labs.NN.MLModelFactory;
import ebtic.labs.NN.NN;
import ebtic.labs.metrics.Accuracy;
import ebtic.labs.metrics.Precision;
import ebtic.labs.metrics.Recall;
import ebtic.labs.objects.Generic;
import ebtic.labs.objects.MyObject;
import ebtic.labs.utils.Matrix;

public class FinalModel {
	
	
	

	
	private static HashMap<String, double[][]> decodeWeights(HashMap<String, EncodedWeight[][]> encodedWeights, HCDecoder decoder)
	{
		
		HashMap<String, double[][]> decoded = new HashMap<String, double[][]>();
		
		Iterator<String> iterator = encodedWeights.keySet().iterator();
		
		while(iterator.hasNext())
		{
			String layer = iterator.next();
			
			EncodedWeight[][] eW = encodedWeights.get(layer);
			int[] shapeA = Matrix.getShape(eW);
			double[][] W= new double[shapeA[0]][shapeA[1]];
			for(int i =0; i< shapeA[0]; i++)
			{
				for(int j=0; j<shapeA[1]; j++)
				{
					double value = 0;
					Iterator<String> keys = eW[i][j].getKeys();
					while(keys.hasNext())
					{
						String hash = keys.next();
						double counter = eW[i][j].getItems(hash);
						//counter = (counter / (double) nodes);
						eW[i][j].replaceWeight(hash, counter);
						
						double aux =  decoder.decode(hash);
						
						aux = aux * counter;
						value = value + aux;
					}
					
					W[i][j] = value;
							
				}
		
			}
			encodedWeights.put(layer, eW);
			decoded.put(layer, W);
			
		}
		return decoded;
		
	}
	
	
	private static HashMap<String, double[][]> finalizeEncWeights(HashMap<String, EncodedWeight[][]> encodedWeights, boolean bias, int nodes, HCDecoder decoder)
	{
		
		HashMap<String, double[][]> output = new HashMap<String, double[][]>();
		HashMap<String, double[][]> modelWeights = decodeWeights(encodedWeights, decoder);
		Iterator iter = modelWeights.keySet().iterator();
		
		
		while(iter.hasNext())
		{
			String layer = (String) iter.next();
			double[][] weights = modelWeights.get(layer);
			
			if(!bias)
			{
				FinalModel.printWeights("W_"+layer+" reading "+nodes, weights);
				output.put(layer, weights);
			}
			else
			{
				FinalModel.printWeights("b_"+layer+" reading "+nodes, weights);
				output.put(layer, weights);
			}
			
			
			output.put(layer, weights);	
			
		}
		
		
		return output;
		
	}
	
	
	public static HashMap<String, double[][]> finalizeWeights(HashMap<String, double[][]> modelWeights, boolean bias, int nodes)
	{
		HashMap<String, double[][]> output = new HashMap<String, double[][]>();
		Iterator iter = modelWeights.keySet().iterator();
		while(iter.hasNext())
		{
			String layer = (String) iter.next();
			double[][] weights = modelWeights.get(layer);
			
			if(!bias)
			{
				FinalModel.printWeights("W_"+layer+" reading ", weights);
				output.put(layer, weights);
			}
			else
			{
				
				FinalModel.printWeights("b_"+layer+" reading ", weights);
				output.put(layer, weights);
			}
			
			
			output.put(layer, weights);	
			
		}
	
		return output;
		
	}
	
	
	public static void main(String[] args) throws IllegalStateException, FileNotFoundException
	{
		Nd4j.setDataType(DataType.DOUBLE);
		String subDir = "HFL";
		String subsubDir = "validation";
		String file = "validation.csv";
		String fileName = "data"+File.separator+subDir+File.separator+subsubDir+File.separator+file;

		
        List<MyObject> beans = Generic.instanciateCSV(fileName);
        
//        beans.forEach(System.out::println);
//        System.out.println(beans.size());

//        Collections.shuffle(beans);
       
        double[][] m = Matrix.getData(beans);
        double[][] labels = Matrix.getLabels(beans);
		
		
        String propFile = "configuration.prop";
		Properties prop = Utils.loadConf(propFile);
        
		String modelClass = Utils.getModelClass(prop); 
		int classes = Utils.getClasses(prop);
		
		double clipping = Utils.getGclipping(prop);
		int[] layers= Utils.getLayers(prop);
		ActivationFunction.Types af = Utils.getActivationFunc(prop);
		String encFile = Utils.getInitWeightsFile(prop);
		boolean hash = Utils.isHashing(prop);
		int nodes = Utils.getNodes(prop);
		boolean withBias = Utils.includesBias(prop);
		
		
		try
        {  
//			MLP mlp = new MLP(layers, af);
			
			
			
			
            // Reading the object from a file
            FileInputStream f = new FileInputStream(encFile);
            ObjectInputStream in = new ObjectInputStream(f);
             
            // Method for deserialization of object
            ArrayList woi=new ArrayList<>();
            woi=(ArrayList)in.readObject();

            HashMap<String, double[][]> weights=null;
            HashMap<String, EncodedWeight[][]> encWeights=null;
            HashMap<String, double[][]> b=null;
            HashMap<String, EncodedWeight[][]> encB=null;
            
            if(woi.size()>0)
            {
            	if(!hash)
            	{
            		weights = (HashMap<String, double[][]>)woi.get(0);
            		weights = finalizeWeights(weights, false, nodes);
            		
		            if(woi.size()>1)
		            {
		            	b = (HashMap<String, double[][]>)woi.get(1);
		            	b = finalizeWeights(b, true, nodes);
		            	
		            	
		            }
		           
            	}
            	
            	else
            	{
            		File enc = new File(Utils.getEncodingFile(prop));
            		HCDecoder decoder = null;
            		if(enc.exists() && !enc.isDirectory()) { 
            		    // do something
            			decoder = new HCDecoder(enc.getAbsolutePath());
            		} 
            		
            		else
            		{
            			System.out.println("Encoding file does not exists, using configuration:");
            			int channels =Utils.getChannels(prop);
            			double min = Utils.getMin(prop);
            			double max = Utils.getMax(prop);
            			System.out.println("Channels: "+channels);
            			System.out.println("Min: "+min);
            			System.out.println("Max: "+max);
            			
            			HCEncoder encoder = new HCEncoder(channels, min, max, true);
            			
            			decoder = new HCDecoder(encoder.getTree());
            		}
            		
            	
            		encWeights = (HashMap<String, EncodedWeight[][]>)woi.get(0);
            		
	            	// Method for deserialization of object
            		weights = finalizeEncWeights(encWeights, false, nodes, decoder);
		            
		            if(woi.size()>1)
		            {
		            	encB = (HashMap<String, EncodedWeight[][]>)woi.get(1);
		            	b = finalizeEncWeights(encB, true, nodes, decoder);
		            	
		            }
		            
            	}
            	
            	
            }
             
            in.close();
            f.close();
             
           
            
            System.out.println("Object has been deserialized ");
            
            
            NN mlp = MLModelFactory.getModelType(modelClass, layers, af, classes, Matrix.getShape(m)[1], clipping);
            
//            f21bc.labs.utils.Util.printWeights(weights);
            
//            mlp.printWeights();
            
            if(b==null) 
            	mlp.instanciate(Matrix.getShape(m)[1], weights);
            else
            	mlp.instanciate(Matrix.getShape(m)[1], weights, b);
 
            
            double[][] results = mlp.predict(m);
            
            System.out.println("Guesses size: "+Matrix.getShape(results)[0]+"x"+Matrix.getShape(results)[1]);
            System.out.println("Labels size: "+Matrix.getShape(labels)[0]+"x"+Matrix.getShape(labels)[1]);
            
            

            
            
            
            mlp.evaluate(labels, results);
            
//            System.out.println(Matrix.toString(results));
            
            
            
            
            
            Accuracy acc = new Accuracy(results, mlp.getLastAF(), classes);
            System.out.println(acc.toString(labels, -10));            
            
            
            Precision pre = new Precision(results, mlp.getLastAF(), classes);
//            System.out.println("Precision for class 0:");
//            System.out.println(pre.toString(labels, 0));
//            System.out.println("Precision for class 1:");
//            System.out.println(pre.toString(labels, 1));
            
            Recall rec = new Recall(results, mlp.getLastAF(), classes);
//            System.out.println("Recall for class 0:");
//            System.out.println(rec.toString(labels, 0));
//            System.out.println("Recall for class 1:");
//            System.out.println(pre.toString(labels, 1));
            
            System.out.println("\n\n");
//            String output = printResults(acc, pre, rec, labels);
            String output = ebtic.labs.utils.Util.metricsTable(pre, rec, acc, labels, af);
            System.out.println(output);
            
        }
         
        catch(IOException ex)
        {
            System.out.println("IOException is caught");
            ex.printStackTrace();
        }
         
        catch(ClassNotFoundException ex)
        {
            System.out.println("ClassNotFoundException is caught");
        }
		catch (MatrixException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	
	public static String printResults(Accuracy acc, Precision pre, Recall rec, double[][] labels)
	{
		if(acc.isBinClass())
			return printResultsBin(acc, pre, rec, labels);
		else
			return printResultsMultiClass(acc, pre, rec, labels);
	}
	
	
	public static String printResultsMultiClass(Accuracy acc, Precision pre, Recall rec, double[][] labels)
	{
		return "-----------------------------------";
	}
	
	public static String printResultsBin(Accuracy acc, Precision pre, Recall rec, double[][] labels)
	{
		double prec_c0_avg = pre.getValue(labels, 0)[1];
        double rec_c0_avg =  rec.getValue(labels, 0)[1];
        int support_0 = (int) rec.getValue(labels, 0)[0];
        double prec_c1_avg = pre.getValue(labels, 1)[1];
        double rec_c1_avg =  rec.getValue(labels, 1)[1];
        int support_1 = (int) rec.getValue(labels, 1)[0];
        double acc_avg =  acc.getValue(labels, 1000)[1];
        int support_acc = support_0+support_1;
        		
        double f1_0 = 2 *((prec_c0_avg*rec_c0_avg/(prec_c0_avg+rec_c0_avg)));
		double f1_1 = 2 *((prec_c1_avg*rec_c1_avg/(prec_c1_avg+rec_c1_avg)));
		
		int cellSize = 10;
		DecimalFormat df = new DecimalFormat("0.0000");
		char[] chars0 = new char[cellSize];
		Arrays.fill(chars0, '-');
		String format = "|%1$-"+Integer.toString(cellSize)+"s|%2$-"+Integer.toString(cellSize)+"s|%3$-"+Integer.toString(cellSize)+"s|%4$-"+Integer.toString(cellSize)+"s|%5$-"+Integer.toString(cellSize)+"s|%6$-"+Integer.toString(cellSize)+"s|\n";
		String[] header = {"Class", "Precision", "Recall", "F1-Score", "Accuracy", "Support"};
		String[] attributeFeet = {new String(chars0), new String(chars0), new String(chars0), new String(chars0), new String(chars0), new String(chars0)};

		String output="";
		output = output+String.format(format, header);
		output = output+String.format(format, attributeFeet);
		
		
		
		
		String[] row1 = {Double.toString(0), df.format(prec_c0_avg), df.format(rec_c0_avg), df.format(f1_0), "",  Integer.toString(support_0)};
		output = output+String.format(format, row1);
		
		
		
		
		
		String[] row2 = {Double.toString(1), df.format(prec_c1_avg), df.format(rec_c1_avg), df.format(f1_1), "",  Integer.toString(support_1)};
		output = output+String.format(format, row2);
		
		
		String[] row3 = {"", "", "", "", df.format(acc_avg),  Integer.toString(support_acc)};
		output = output+String.format(format, row3);
		
		String[] row4 = {"macro_avg",df.format((prec_c0_avg+prec_c1_avg)/2) , df.format((rec_c0_avg+rec_c1_avg)/2), df.format((f1_0+f1_1)/2), "",  ""};
		output = output+String.format(format, row4);
		
		
		double weight_0 = (Double.valueOf(support_0)/Double.valueOf(support_acc));
		double weight_1 = (Double.valueOf(support_1)/Double.valueOf(support_acc));
		
//		System.out.println(weight_0);
//		System.out.println(weight_1);
		
		System.out.println("\n");
		String[] row5 = {"weigh_avg",df.format((prec_c0_avg*weight_0)+(prec_c1_avg*weight_1)) , df.format((rec_c0_avg*weight_0)+(rec_c1_avg*weight_1)), df.format((f1_0*weight_0)+(f1_1*weight_1)), "",  ""};
		output = output+String.format(format, row5);
		output = output+String.format(format, attributeFeet);
		
		return output;
	}
	
	
	public static void printWeights(String layer, double[][] aux)
	{
		int i = 1;
		int j = 0;
		int[] shape = Matrix.getShape(aux);
		
		if((i < shape[0])&&(j < shape[1]))
			System.out.println("Layer: "+layer+" shape: "+Matrix.printShape(aux)+" "+i+"x"+j+" value: "+aux[i][j]);	
		else
			System.out.println("Layer: "+layer+" shape: "+Matrix.printShape(aux)+" "+i+"x"+j+" no value");
	}
	
	public static void printEncWeights(String layer, EncodedWeight[][] aux)
	{
		int i = 1;
		int j = 0;
		int[] shape = Matrix.getShape(aux);
		
		if((i < shape[0])&&(j < shape[1]))
			System.out.println("Layer: "+layer+" shape: "+Matrix.printShape(aux)+" "+i+"x"+j+" value: "+aux[i][j].toString());	
		else
			System.out.println("Layer: "+layer+" shape: "+Matrix.printShape(aux)+" "+i+"x"+j+" no value");
	}
	
}
