package ebtic.labs.LF;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.AF.ReLU;
import ebtic.labs.AF.Sigmoid;
import ebtic.labs.AF.Tanh;
import ebtic.labs.AF.ActivationFunction.Types;
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
