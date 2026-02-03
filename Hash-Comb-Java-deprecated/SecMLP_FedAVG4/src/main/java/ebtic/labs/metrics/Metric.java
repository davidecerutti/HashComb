package ebtic.labs.metrics;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.AF.ReLU;
import ebtic.labs.AF.Sigmoid;
import ebtic.labs.AF.Tanh;
/**
 * abstract class defining Metric to evaluate the performance of MLP
 *
 */
public abstract class Metric {

	private ActivationFunction.Types af;
	
	public Metric(ActivationFunction.Types af)
	{
		this.af = af;
	}
	
	
	public int predict(double value)
	{
		int prediction = -10000;
		switch (this.af) {
	    
		case SIGMOID   : 
	    	if(value<0.5)
			prediction = 0;
		else
			prediction = 1; break;
	    
	    
	    case TANH :  
	    	if(value<0.5)
			prediction = 0;
		else
			prediction = 1; break;
	    
	    
	    
	    case ReLU   : 
	    	if(value<0.5)
			prediction = 0;
		else
			prediction = 1; break;
	    
		}
		
		return prediction;
		
	}
	
	public abstract double[] getValue(double[][] labels, int myclass);
	
	public abstract String toString(double[][] labels, int myclass);
}
