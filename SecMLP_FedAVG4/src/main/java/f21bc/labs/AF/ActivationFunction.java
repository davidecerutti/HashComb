package f21bc.labs.AF;

/**
 * abstract class defining Activation Functions
 * @author maurizio
 *
 */
public abstract class ActivationFunction {

	
	public enum Types {
	    SIGMOID,
	    TANH,
	    ReLU,
	    SoftMAX
	}
	
	public abstract double[][] exec(double[][] Z);
	
	public abstract double[][] exec_derivative(double[][] A);
	
	
	public abstract Types getType();
	
	/*
	 * 
	 */
	public static ActivationFunction instanciate(Types type)
	{
		
		ActivationFunction function=null;
		
		switch (type) {
	    case SIGMOID   : function = new Sigmoid(); break;
	    case TANH :   function = new Tanh(); break;
	    case ReLU   : function = new ReLU(); break;
	    case SoftMAX : function = new SoftMAX(); break;
	    
		}
		return function;
		
	}
}
