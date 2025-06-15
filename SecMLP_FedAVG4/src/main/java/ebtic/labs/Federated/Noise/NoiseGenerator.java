package ebtic.labs.Federated.Noise;

public abstract class NoiseGenerator {

	
	public abstract double generate();
	
	public abstract double[][] addNoise(double[][] weights);
	
	public abstract void setMean(double mean);
	public abstract void setVariance(int steps, int dataset, double lr, double minibatch, double clipping);
	
}
