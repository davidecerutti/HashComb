package ebtic.labs.metrics;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.utils.Matrix;
/**
 * This class calculate the Accuracy of each epoch 
 *
 */
public class Accuracy extends Metric{
	
	private double[][] results;
	private int classes =0;
	
	public Accuracy(double[][] results, ActivationFunction.Types af, int classes)
	{
		super(af);
		this.results = results;
		this.classes = classes;
		
		if(classes<3)  //binary classification
		{
			if(results[0].length==1)  // values on the first column
				this.results = Matrix.transpose(results);
		}
		else   //multiclass
		{
			int[] shape = Matrix.getShape(results);
			if((shape[0]!=shape[1])&&(shape[1]==classes))
				this.results = Matrix.transpose(results);
		}
		
	}
	
	public double[] getValue(double[][] labels, int myClass)
	{
		if(this.classes>2)
			return this.getValueMultiClass(labels);
		else
			return this.getValueBinClass(labels);

	}

	
	private double[] getValueBinClass(double[][] labels)
	{
		
		
		double count = 0;
		for(int j=0; j<this.results[0].length; j++)
		{
			int prediction = predict(this.results[0][j]);
//			if(this.results[0][j]<0.5)
//				prediction = 0;
//			else
//				prediction = 1;
			if(prediction==labels[0][j])
				count++;
		}
		
		double accuracy = ((count) / labels[0].length);
//		System.out.println("("+count+") / "+labels[0].length+" = "+accuracy);
		
		double[] out = {count, accuracy};
		return out;
	}

	
	private double[] getValueMultiClass(double[][] labels)
	{
		double count = 0;
		for(int col=0; col<this.results[0].length; col++)
		{
			
			int numRows = results.length;
	        int numCols = results[0].length;
			// Extract the entire column
            double[] columnArray = new double[results.length];
            
            for (int row = 0; row < numRows; row++) {
                columnArray[row] = results[row][col];
            }
			
            int prediction = ebtic.labs.AF.SoftMAX.argMax(columnArray);
//			if(this.results[0][j]<0.5)
//				prediction = 0;
//			else
//				prediction = 1;
			if(prediction==labels[0][col])
				count++;
		}
		
		double accuracy = ((count) / labels[0].length);
//		System.out.println("("+count+") / "+labels[0].length+" = "+accuracy);
		
		double[] out = {count, accuracy};
		return out;
	}


	public boolean isBinClass()
	{
		return (this.classes<3);
	}
	

	public String toString(double[][] labels, int myClass)
	{
		
		
		String out = "";
		double[] acc = this.getValue(labels, 1000);
		out = out + "Correct predictions: "+acc[0]+"\n";
		out = out + "Accuracy: "+acc[1]*100+"%\n";
		out = out + "Support: "+Matrix.getShape(this.results)[1]+"\n";
		
		return out;
	}
	
	
	public int getClassesN()
	{
		return this.classes;
	}
}
