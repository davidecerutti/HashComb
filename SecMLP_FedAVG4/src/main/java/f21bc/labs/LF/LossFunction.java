package f21bc.labs.LF;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.AF.ReLU;
import f21bc.labs.AF.Sigmoid;
import f21bc.labs.AF.Tanh;
import f21bc.labs.AF.ActivationFunction.Types;
/**
 * abstract class defining Loss Functions
 *
 */
public abstract class LossFunction {
	
	public enum Types {
	    CE,
	    MSE
	    
	}

	public abstract double exec(double label, double value);
	
	public abstract double[][] exec(double[][] labels, double[][] values);
	
	
	public static LossFunction instanciate(Types type)
	{
		
		LossFunction function=null;
		
		switch (type) {
	    case CE  : function = new CrossEntropy(); break;
	    case MSE : function = new MeanSquaredError(); break;
		}
		
		
			
		
		return function;
		
	}
}
