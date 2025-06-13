package f21bc.labs.Federated.Noise;

import java.util.Random;

import f21bc.labs.utils.Matrix;

public class GaussianNoise extends NoiseGenerator{

    public double mean;
    public double variance;
    
    //smaller epsilon represents stronger privacy protection level
    private double epsilon=2;
    
    //δ ∈ [0, 1] stands for the probability to break the (ǫ,0)-DP.
    private double delta = Math.pow(10,-3);
    		

    public GaussianNoise(double mean, double variance) {
     this.mean = mean;
     this.variance = variance;
     System.out.println("Instantiating a new Gaussian Noise generator....");
    }
    
    public GaussianNoise() {
        this.mean = 0;
        this.variance = 1;
        System.out.println("Instantiating a new (default) Gaussian Noise generator....");
       }
    
    
    
    public void configure(double mean, double variance)
    {
    	 this.mean = mean;
         this.variance = variance;
    }
    
    public double generate()
    {
    	
    	return (mean + new Random().nextGaussian() * variance);
    }
    
  
    
	@Override
	public double[][] addNoise(double[][] weights) {
		// TODO Auto-generated method stub
		return Matrix.addGaussianNoise(this, weights);
	}
    
    public static void main(String[] args)
    {
    	
    	double value = 0.5463784849576;
    	
    	GaussianNoise generator = new GaussianNoise(0,2);
    	
    	generator.setMean(0);
    	generator.setVariance(1000, 1000, 0.05, 0.25, 1);
    	double noise= generator.generate(); 
    	
    	System.out.println("Value: "+value+"     gauss: "+noise+"     new value: "+(noise+value));
    }

	@Override
	public void setVariance(int steps, int dataset, double lr, double minibatch, double clipping) {
		// TODO Auto-generated method stub
		double q = ((double)steps*minibatch);
//		double q = ((double)steps/dataset);
		//q = 0.25;
		
//		double C = 2.0;
//		double C = 0.87;

		double C = 4*(double)steps*clipping*lr; 
//		C=0.02;
		
		
		double log = Math.log((1.25*q)/delta);
		//double out =  Math.sqrt(((2*(Math.log((1.25)/delta)))/Math.pow(epsilon,2))); 
		//double out =  ((2*(Math.log((1.25)/delta)))/Math.pow(epsilon,2)); 
		double out =  ((8*Math.pow(q,2)*Math.pow(C,2)*log)/Math.pow(epsilon,2)); 
		
		
		
		System.out.println("Q: "+(double)steps);
		System.out.println("q: "+q);
		System.out.println("C: "+C);
		System.out.println("ds: "+dataset);
		System.out.println("batch size: "+minibatch);
		System.out.println("delta: "+delta);
		System.out.println("epsilon: "+epsilon);
		System.out.println("LR: "+lr);
		System.out.println("calculated   variance= "+out);
		this.variance=out;
	}

	@Override
	public void setMean(double mean) {
		// TODO Auto-generated method stub
		this.mean = mean;
	}


	

}