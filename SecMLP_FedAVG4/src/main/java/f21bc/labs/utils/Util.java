package f21bc.labs.utils;

import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Formatter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.NN.Layer;
import f21bc.labs.metrics.Accuracy;
import f21bc.labs.metrics.History;
import f21bc.labs.metrics.Precision;
import f21bc.labs.metrics.Recall;
/**
 * The class contains some utilities used for the implementation
 * @author maurizio
 *
 */
public class Util {
	
	/**
	 * Read a CSV file and return the lines as an array of string
	 * @param fileName the input file
	 * @return the lines of the file as an array of strings
	 * @throws IOException
	 * @throws CsvException
	 */
	public static List<String[]> readCSV(String fileName) throws IOException, CsvException {

        //fileName = "..\\w2\\archive\\diabetes.csv";
        List<String[]> output = null;
        try (CSVReader reader = new CSVReader(new FileReader(fileName))) {
            output = reader.readAll();
            output.forEach(x -> System.out.println(Arrays.toString(x)));
        }
		return output;

    }

	/**
	 * The method generate a table containing the metrics 
	 * @param precision the Precision object
	 * @param recall the Recall object
	 * @param accuracy the Accuracy object
	 * @param labels the labels against with to evaluate the model
	 * @return the string representing the table with the results
	 */
	
	public static String metricsTable(Precision precision, Recall recall, Accuracy accuracy, double[][] labels, ActivationFunction.Types af)
	{
		if(accuracy.isBinClass())
			return metricsTableBin(precision, recall, accuracy, labels, af);
		else
			return metricsTableMulti(precision, recall, accuracy, labels, af);
	}
	
	
	public static String metricsTableMulti(Precision precision, Recall recall, Accuracy accuracy, double[][] labels, ActivationFunction.Types af)
	{
		af = ActivationFunction.Types.SoftMAX;
		int cellSize = 10;
		//int[] awards = this.competitors.scoreFrequency();
		String format = "|%1$-"+Integer.toString(cellSize)+"s|%2$-"+Integer.toString(cellSize)+"s|%3$-"+Integer.toString(cellSize)+"s|%4$-"+Integer.toString(cellSize)+"s|%5$-"+Integer.toString(cellSize)+"s|%6$-"+Integer.toString(cellSize)+"s|\n";
		DecimalFormat df = new DecimalFormat("0.0000");
		char[] chars0 = new char[cellSize];
		Arrays.fill(chars0, '-');


		String[] header = {"Class", "Precision", "Recall", "F1-Score", "Accuracy", "Support"};
		String[] attributeFeet = {new String(chars0),new String(chars0), new String(chars0), new String(chars0), new String(chars0), new String(chars0)};

		String output="";
		output = output+String.format(format, header);
		output = output+String.format(format, attributeFeet);
		

		int[] classes = null;
		
		if(af.equals(ActivationFunction.Types.SIGMOID))
		{
			classes = new int[2];
			classes[0] = 0;
			classes[1] = 1;
		}
		
		else if(af.equals(ActivationFunction.Types.SoftMAX))
		{
			classes = new int[accuracy.getClassesN()];
			for(int i=0; i<classes.length; i++)
				classes[i] = i;
		}
		
		
		
		
		double tot_R = 0;
		double tot_P =0;
		double tot_f1 = 0;
		
		for(int i=0; i<classes.length; i++)
		{
			int cl = classes[i];
			double[] outP = precision.getValue(labels, cl);
			double[] outR = recall.getValue(labels, cl);
			double f1 = 2 *((outP[1]*outR[1]/(outP[1]+outR[1])));
			
			
			tot_R = tot_R + outR[1];
			tot_P = tot_P + outP[1];
			tot_f1 = tot_f1 + f1;
			
			
			String[] row = {Double.toString(cl), df.format(outP[1]), df.format(outR[1]), df.format(f1), "",  df.format(outP[0])};
			
			output = output+String.format(format, row);
		}
		
		
		
		
		double[] outA = accuracy.getValue(labels, 1000);
		String[] row3 = {"", "", "", "", df.format(outA[1]),  Integer.toString(labels[0].length)};
		output = output+String.format(format, row3);
		
		
		String[] row4 = {"macro_avg",df.format((tot_P)/classes.length) , df.format((tot_R)/classes.length), df.format((tot_f1)/classes.length), "",  ""};
		output = output+String.format(format, row4);
		
		output = output+String.format(format, attributeFeet);
		return output;

	}
	
	public static String metricsTableBin(Precision precision, Recall recall, Accuracy accuracy, double[][] labels, ActivationFunction.Types af)
	{
		int cellSize = 10;
		//int[] awards = this.competitors.scoreFrequency();
		String format = "|%1$-"+Integer.toString(cellSize)+"s|%2$-"+Integer.toString(cellSize)+"s|%3$-"+Integer.toString(cellSize)+"s|%4$-"+Integer.toString(cellSize)+"s|%5$-"+Integer.toString(cellSize)+"s|%6$-"+Integer.toString(cellSize)+"s|\n";
		DecimalFormat df = new DecimalFormat("0.0000");
		char[] chars0 = new char[cellSize];
		Arrays.fill(chars0, '-');


		String[] header = {"Class", "Precision", "Recall", "F1-Score", "Accuracy", "Support"};
		String[] attributeFeet = {new String(chars0),new String(chars0), new String(chars0), new String(chars0), new String(chars0), new String(chars0)};

		String output="";
		output = output+String.format(format, header);
		output = output+String.format(format, attributeFeet);
		
//		Precision precision = new Precision(results);
//		Recall recall = new Recall(results);
//		Accuracy accuracy = new Accuracy(results);
		
		int myClass0 = 0;
		int myClass1 = 0;
		
		if(af.equals(ActivationFunction.Types.SIGMOID))
		{
			myClass0 = 0;
			myClass1 = 1;
		}
		
		else if(af.equals(ActivationFunction.Types.TANH))
		{
			myClass0 = 0;
			myClass1 = 1;
		}
		
		else if(af.equals(ActivationFunction.Types.ReLU))
		{
			myClass0 = 0;
			myClass1 = 1;
		}
		
		
		
		
		double[] outP_0 = precision.getValue(labels, myClass0);
		double[] outR_0 = recall.getValue(labels, myClass0);
		double f1_0 = 2 *((outP_0[1]*outR_0[1]/(outP_0[1]+outR_0[1])));
		
		
		
		String[] row1 = {Double.toString(myClass0), df.format(outP_0[1]), df.format(outR_0[1]), df.format(f1_0), "",  df.format(outP_0[0])};
		output = output+String.format(format, row1);
		
		
		
		double[] outP_1 = precision.getValue(labels, myClass1);
		double[] outR_1 = recall.getValue(labels, myClass1);
		
		double f1_1 = 2 *((outP_1[1]*outR_1[1]/(outP_1[1]+outR_1[1])));
		
		String[] row2 = {Double.toString(myClass1), df.format(outP_1[1]), df.format(outR_1[1]), df.format(f1_1), "",  df.format(outP_1[0])};
		output = output+String.format(format, row2);
		
		double[] outA = accuracy.getValue(labels, 1000);
		String[] row3 = {"", "", "", "", df.format(outA[1]),  Integer.toString(labels[0].length)};
		output = output+String.format(format, row3);
		
		
		String[] row4 = {"macro_avg",df.format((outP_0[1]+outP_1[1])/2) , df.format((outR_0[1]+outR_1[1])/2), df.format((f1_0+f1_1)/2), "",  ""};
		output = output+String.format(format, row4);
		
		output = output+String.format(format, attributeFeet);
		return output;

	}

	/**
	 * The method generate a table containing the metrics as average after a cross validation training
	 * @param cv_hystory the array containing all the K-fold trainings
	 * @param type returns the metric either of the training or the evaluation 
	 * @return the string representing the table with the results
	 */
	public static String metricsTable(ArrayList<History> cv_hystory, History.Type type)
	{
		int cellSize = 10;
		//int[] awards = this.competitors.scoreFrequency();
		String format = "|%1$-"+Integer.toString(cellSize)+"s|%2$-"+Integer.toString(cellSize)+"s|%3$-"+Integer.toString(cellSize)+"s|%4$-"+Integer.toString(cellSize)+"s|%5$-"+Integer.toString(cellSize)+"s|%6$-"+Integer.toString(cellSize)+"s|\n";
		DecimalFormat df = new DecimalFormat("0.0000");
		char[] chars0 = new char[cellSize];
		Arrays.fill(chars0, '-');


		String[] header = {"Class", "Precision", "Recall", "F1-Score", "Accuracy", "Support"};
		String[] attributeFeet = {new String(chars0), new String(chars0), new String(chars0), new String(chars0), new String(chars0), new String(chars0)};

		String output="";
		output = output+String.format(format, header);
		output = output+String.format(format, attributeFeet);
		
//		Precision precision = new Precision(results);
//		Recall recall = new Recall(results);
//		Accuracy accuracy = new Accuracy(results);
		
		
		
		int myClass = 0;
		
		double acc_avg=0;
		double prec_c0_avg=0;
		double prec_c1_avg=0;
		double rec_c0_avg=0;
		double rec_c1_avg=0;
		
		for(int i=0; i<cv_hystory.size(); i++)
		{
			History history = cv_hystory.get(i);
			double[][] labels;
			
			if(type.equals(History.Type.TRAINING))
				labels = history.getTrainingLabels();
			else
				labels = history.getTestingLabels();
			
			
			
			Precision precision = history.getPrecision(type);
			Recall recall = history.getRecall(type);
			Accuracy accuracy= history.getAccuracy(type);
			
			prec_c0_avg = prec_c0_avg + precision.getValue(labels, 0)[1];
			rec_c0_avg = rec_c0_avg + recall.getValue(labels, 0)[1];
			prec_c1_avg = prec_c1_avg + precision.getValue(labels, 1)[1];
			rec_c1_avg = rec_c1_avg + recall.getValue(labels, 1)[1];
			acc_avg = acc_avg + accuracy.getValue(labels, 1000)[1];
		}
		
		prec_c0_avg = prec_c0_avg / cv_hystory.size();
		rec_c0_avg =  rec_c0_avg / cv_hystory.size();
		prec_c1_avg = prec_c1_avg / cv_hystory.size();
		rec_c1_avg = rec_c1_avg / cv_hystory.size();
		acc_avg = acc_avg / cv_hystory.size();
		
		double f1_0 = 2 *((prec_c0_avg*rec_c0_avg/(prec_c0_avg+rec_c0_avg)));
		double f1_1 = 2 *((prec_c1_avg*rec_c1_avg/(prec_c1_avg+rec_c1_avg)));
		
		String[] row1 = {Double.toString(0), df.format(prec_c0_avg), df.format(rec_c0_avg), df.format(f1_0), "",  ""};
		output = output+String.format(format, row1);
		
		
		
		
		
		String[] row2 = {Double.toString(1), df.format(prec_c1_avg), df.format(rec_c1_avg), df.format(f1_1), "",  ""};
		output = output+String.format(format, row2);
		
		
		String[] row3 = {"", "", "", "", df.format(acc_avg),  ""};
		output = output+String.format(format, row3);
		
		String[] row4 = {"macro_avg",df.format((prec_c0_avg+prec_c1_avg)/2) , df.format((rec_c0_avg+rec_c1_avg)/2), df.format((f1_0+f1_1)/2), "",  ""};
		output = output+String.format(format, row4);
		
		output = output+String.format(format, attributeFeet);
		return output;

	}

	
	
	
	public static void printLayersWeight(ArrayList<Layer> layers, int r, int c)
	{
		

	       for(int i = 0; i<layers.size(); i++)
	       {
	    	   double[][] wMatrix = layers.get(i).get_W();
	    	   
	    	   System.out.println("-Layer_"+i+" W["+r+","+c+"] :"+wMatrix[r][c]);

	    	   if(i<layers.size()-1)
	    	   {
	    		   
		    	   double[][] bMatrix = layers.get(i).get_b();
		    	   System.out.println("-Layer_"+i+" b["+r+","+c+"] :"+bMatrix[r][c]); 
	    	   }
	    	  
	    	   System.out.println("---------");
	       }
		
		
	}

	
	
	
	public static void printWeights(HashMap <String, double[][]> weights )
	{
		Iterator iter = weights.keySet().iterator();
		
	       while(iter.hasNext())
	       {
	    	   String key = (String) iter.next();

	    	   double[][] wMatrix = weights.get(key);
	    	   
	  
//	    	   int[] shape = Matrix.getShape(out);
//	    	   System.out.println("Layer_"+i+" shape: "+shape[0]+" x "+shape[1]);
	    	   System.out.println("-Layer_"+key);
	    	   System.out.println("W size "+wMatrix.length+"x"+wMatrix[0].length);
	    	   
	    	   System.out.println("---------");
	       }
		
	}


	
    // Concrete method to calculate minMax
    public static double[] calculateMinMax(ArrayList<Layer> layers) {
        if (layers == null || layers.isEmpty()) {
            throw new IllegalArgumentException("The input list is null or empty");
        }

        // Initialize min and max values with extreme values
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        // Loop through each matrix in the list
        for (Layer l : layers) {
        	
        	double[][] W = l.get_W();
        	double[][] b = l.get_b();
        	
        	 for (int j = 0; j < b.length; j++) {
                 // Update min and max
                 if (b[j][0] < min) {
                     min = b[j][0];
                 }
                 if (b[j][0] > max) {
                     max = b[j][0];
                 }
             }
        	
            for (int i = 0; i < W.length; i++) {
                for (int j = 0; j < W[i].length; j++) {
                    // Update min and max
                    if (W[i][j] < min) {
                        min = W[i][j];
                    }
                    if (W[i][j] > max) {
                        max = W[i][j];
                    }
                }
            }
        }

        // Return the min and max as a double array
        return new double[]{min, max};
    }
	
	
	public static void main(String[] args)
	{
		try {
			readCSV("..\\w2\\archive\\diabetes.csv");
		} catch (IOException | CsvException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
