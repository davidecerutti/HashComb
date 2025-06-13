package f21bc.labs.metrics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.AF.ReLU;
import f21bc.labs.AF.Sigmoid;
import f21bc.labs.AF.Tanh;
import f21bc.labs.utils.Matrix;

public class History {
	
	
	public enum Type {
	    TRAINING,
	    VALIDATION
	}
	
	private HashMap<Integer, double[][]> results;
	private int epochs;
	private int counter = 0;
	private double[][] trainingLabels;
	private double[][] testingLabels;
	private ActivationFunction.Types af;
	
	private HashMap<String, double[][]> weights;
	private HashMap<String, double[][]> bias;
	
	public History(int epochs, ActivationFunction.Types af )
	{
		this.epochs = epochs;
		this.results = new HashMap<Integer, double[][]>();
		this.af = af;
		this.weights = new HashMap<String, double[][]>();
		this.bias = new HashMap<String, double[][]>();
	}
	
	
	public void setTrainingLabels(double[][] labels)
	{
		this.trainingLabels = labels;
		
	}
	
	public void setTestingLabels(double[][] labels)
	{
		
		this.testingLabels = labels;
		
	}
	
	public double[][] getTrainingLabels()
	{
		return this.trainingLabels;
	}
	
	public double[][] getTestingLabels()
	{
		return this.testingLabels;
	}
	
	public int getEpochs()
	{
		return this.counter;
	}
	
	public void addTraining(double[][] result)
	{
		results.put(counter, result);
		counter++;
	}
	
	public void addValidation(double[][] result)
	{
		results.put(-1, result);
		
	}
	
	public Accuracy getAccuracy(int epoch)
	{
		if(results.containsKey(epoch))
		{
			double[][] result = this.results.get(epoch);
			Accuracy acc = new Accuracy(result, af, result.length);
			return acc;
		}
		return null;
	}
	
	public Accuracy getAccuracy(History.Type type)
	{
		int index = -2;
		switch (type) {
	    case TRAINING   : index = counter-1; break;
	    case VALIDATION :   index = -1; break;
	    
		}
		if(index!=-2)
			return this.getAccuracy(index);
		else return null;
	}

	
	public Precision getPrecision(int epoch)
	{
		if(results.containsKey(epoch))
		{
			double[][] result = this.results.get(epoch);
			Precision prec = new Precision(result, this.af,result.length);
			return prec;
		}
		return null;
	}
	
	public Precision getPrecision(History.Type type)
	{
		int index = -2;
		switch (type) {
	    case TRAINING   : index = counter-1; break;
	    case VALIDATION :   index = -1; break;
	    
		}
		if(index!=-2)
			return this.getPrecision(index);
		else return null;
	}


	public Recall getRecall(int epoch)
	{
		if(results.containsKey(epoch))
		{
			double[][] result = this.results.get(epoch);
			Recall rec = new Recall(result, this.af, result.length);
			return rec;
		}
		return null;
	}
	
	public Recall getRecall(History.Type type)
	{
		int index = -2;
		switch (type) {
	    case TRAINING   : index = counter-1; break;
	    case VALIDATION :   index = -1; break;
	    
		}
		if(index!=-2)
			return this.getRecall(index);
		else return null;
	}

	public void setWeights(String layer, double[][] weights)
	{
		this.weights.put(layer, weights);
	}
	
	
	public void setBias(String layer, double[][] bias)
	{
		this.bias.put(layer, bias);
	}
	
	
	public double[][] getWeights(String layer)
	{
		return this.weights.get(layer);
	}
	
	
	public double[][] getBias(String layer)
	{
		return this.bias.get(layer);
	}
	
	public HashMap<String, double[][]> getAllWeights()
	{
		Iterator iter =  this.weights.keySet().iterator();
		while(iter.hasNext())
		{
			String k = (String) iter.next();
			double[][] m = this.weights.get(k);
			int[] shape = Matrix.getShape(m);
			int totsize = shape[0] * shape[1];
			System.out.println("Layer (W) "+k+"   :"+shape[0]+"x"+shape[1]+"  size: "+totsize);
		}
			
		return this.weights;
	}
	
	public HashMap<String, double[][]> getAllBias()
	{
		// just to print the bias
		Iterator iter =  this.bias.keySet().iterator();
		while(iter.hasNext())
		{
			String k = (String) iter.next();
			double[][] b = this.bias.get(k);
			int[] shape = Matrix.getShape(b);
			int totsize = shape[0] * shape[1];
			System.out.println("Layer (b) "+k+"   :"+shape[0]+"x"+shape[1]+"  size: "+totsize);
		}
			
		return this.bias;
	}
}
