package f21bc.labs.NN;
/**
 * THe class implement the early stop, when the model converge and the learning process is terminated
 * @author maurizio
 *
 */
public class EarlyStopping {

	// Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
	private double min_delta;
	
	// Number of epochs with no improvement after which training will be stopped.
	private int  patience;
	
	private int count;
	
	private double best = 10;
	
	/**
	 * 
	 * @param min_delta the delta value of the Loss between 2 consequtive observations
	 * @param patience number of iteration without learning before stopping the process
	 */
	public EarlyStopping(double min_delta, int patience)
	{
		this.min_delta = min_delta;
		this.patience = patience;
		this.count = 0;
	}

	/**
	 * the function is call to check when is necessary to stop
	 * @param loss_value the value loss at the current iteration
	 * @return 
	 */
//	public boolean stop(double loss_value)
//	{
//		System.out.println("Loss: "+loss_value+"   Delta: "+min_delta+"   count: "+this.count);
//		if(loss_value>min_delta)
//			this.count = 0;
//		else
//		{
//			this.count++;
//			if(this.count>this.patience)
//				return true;
//			
//		}
//		
//		return false;
//	}


	public boolean stop(double loss_value)
	{
		double limit = best-min_delta;
		
//		System.out.println("Loss: "+loss_value+"   Best: "+limit+"   count: "+this.count);
		if(loss_value<(limit))
		{
			best = loss_value;
			this.count = 0;
		}
			
		else
		{
			this.count++;
			if(this.count>this.patience)
			{
				System.out.println("Stopping Loss: "+loss_value+"   Best: "+limit+"   count: "+this.count);
				return true;
			}
				
			
		}
		
		return false;
	}

	
	
}
