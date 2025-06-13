package f21bc.labs.tests;

import f21bc.labs.Federated.Noise.GaussianNoise;

public class NoiseTest {
	
	
	public static void main(String[] args)
	{
		GaussianNoise noiser = new GaussianNoise();
		
		noiser.setMean(0);
		noiser.setVariance(1000, 4933, 0.05, 0.25, 1);
	}

}
