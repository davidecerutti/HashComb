package ebtic.labs.metrics;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.utils.Matrix;

/**
 * This class calculate the Recall of each epoch 
 *
 */
public class Recall extends Metric {
	
	private double[][] results;
	private int classes =0;
	
	
	public Recall(double[][] results, ActivationFunction.Types af, int classes)
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
			return this.getValueMultiClass(labels, myClass);
		else
			return this.getValueBinClass(labels, myClass);

	}
	
	private double[] getValueMultiClass(double[][] labels, int myClass)
	{
		double support= 0;
		double TP = 0;
		double FN = 0;
//		System.out.println("Total support Recall: "+labels[0].length);
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
			if(labels[0][col]==myClass)
			{
				if(prediction==labels[0][col])
					TP++;
				else
					FN++;
				support++;
			}
			
			
		}
		double aux = (TP + FN);
		double recall = ((TP) / aux);
		System.out.println("Recall: "+"("+TP+") / ("+aux+") = "+recall);
		double[] out = {support, recall};
		return out;
	}
	
	private double[] getValueBinClass(double[][] labels, int myClass)
	{
		double support= 0;
		double TP = 0;
		double FN = 0;
//		System.out.println("Total support Recall: "+labels[0].length);
		for(int j=0; j<this.results[0].length; j++)
		{
			int prediction = predict(this.results[0][j]);;
//			if(this.results[0][j]<0.5)
//				prediction = 0;
//			else
//				prediction = 1;
			if(labels[0][j]==myClass)
			{
				if(prediction==labels[0][j])
					TP++;
				else
					FN++;
				support++;
			}
			
			
		}
		double aux = (TP + FN);
		double recall = ((TP) / aux);
		System.out.println("Recall: "+"("+TP+") / ("+aux+") = "+recall);
		double[] out = {support, recall};
		return out;
	}
	

	public String toString(double[][] labels, int myClass)
	{
		String out = "";
		double[] prec = this.getValue(labels, myClass);
		out = out + "Support: "+prec[0]+"\n";
		//out = out + "Recall: "+prec[1]+"\n";
		
		return out;
	}
	
}
