package ebtic.labs.Federated.threads;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import ebtic.labs.AF.ActivationFunction.Types;
import ebtic.labs.Exceptions.ConfigurationException;
import ebtic.labs.Exceptions.MatrixException;
import ebtic.labs.Federated.communications.Client;
import ebtic.labs.Federated.hashComb.HCDecoder;
import ebtic.labs.Federated.hashComb.HCEncoder;
import ebtic.labs.Federated.utils.EncodedWeight;
import ebtic.labs.NN.EarlyStopping;
import ebtic.labs.metrics.Accuracy;
import ebtic.labs.metrics.History;
import ebtic.labs.metrics.Precision;
import ebtic.labs.metrics.Recall;
import ebtic.labs.utils.Matrix;
import ebtic.labs.utils.PlotTraining;
import ebtic.labs.utils.Util;
import khalifa.ebtic.security.hashcomb.tree.Tree;

public class EncodedNodeThread extends NodeThread{

	private HCEncoder encoder;
	private HCDecoder decoder;
	
	private int channels=4;
	private double min = -1;
	private double max = 1;
	private boolean hashed = true;
	
	
	public EncodedNodeThread(int iteration, int node, String filenameString, int epochs, double learning_rate,
			int[] layers, Types activation_function, ebtic.labs.LF.LossFunction.Types loss_function, EarlyStopping ea, boolean bias, String modelClass,  double batchSize, double clipping)
			throws ConfigurationException, IllegalStateException, FileNotFoundException {
		
		
		super(iteration, node, filenameString, epochs, learning_rate, layers, activation_function, loss_function, ea, bias, modelClass, batchSize, clipping);
		// TODO Auto-generated constructor stub
		 
		this.encoder = new HCEncoder(this.channels, this.min, this.max, this.hashed);
		this.decoder = new HCDecoder("hashmap.ser");
	}
	
	
	public void configure(HCDecoder decoder)
	{
		this.decoder = decoder;
	}
	
	
	public void configure(String filename)
	{
		this.decoder = new HCDecoder(filename);
	}
	
	public void configure(Tree root)
	{
		this.decoder = new HCDecoder(root);
	}
	
	
	public void configure(int channels, double min, double max, boolean hashed)
	{
		this.channels = channels;
		this.min = min;
		this.max= max;
		this.hashed = hashed;
		this.encoder = new HCEncoder(this.channels, this.min, this.max, this.hashed);
	}
	
	
	public void configure(HCEncoder encoder)
	{
		this.encoder = encoder;
	}
	
	
	public void store(String filename)
	{
		this.encoder.store(filename);
	}
	
	

	public void initializeNN(HashMap encodedWeights , HashMap encodedBias)
	{
		
		HashMap<String, double[][]> weights = this.finalizeEncWeights((HashMap<String, EncodedWeight[][]>)encodedWeights, false);
		HashMap<String, double[][]> bias = this.finalizeEncWeights((HashMap<String, EncodedWeight[][]>)encodedBias, true);
		
		mlp.instanciate(weights, bias);
	}
	
	
	public void initializeNN(HashMap encodedWeights)
	{
		
		HashMap<String, double[][]> weights = this.finalizeEncWeights((HashMap<String, EncodedWeight[][]>)encodedWeights, false);
		
		
		mlp.instanciate(weights);
	}
	
	
	private HashMap<String, double[][]> finalizeEncWeights(HashMap<String, EncodedWeight[][]> encodedWeights, boolean bias)
	{
		
		HashMap<String, double[][]> output = new HashMap<String, double[][]>();
		HashMap<String, double[][]> modelWeights = decodeWeights(encodedWeights);
		Iterator iter = modelWeights.keySet().iterator();
		
		
		while(iter.hasNext())
		{
			String layer = (String) iter.next();
			double[][] weights = modelWeights.get(layer);
			EncodedWeight[][] Eweights = encodedWeights.get(layer);
			
			if(!bias)
			{
				FinalModel.printEncWeights("W_"+layer+" Received encoded ", Eweights);
				FinalModel.printWeights("W_"+layer+" After decoding ", weights);
				output.put(layer, weights);
			}
			else
			{
				FinalModel.printEncWeights("b_"+layer+" Received encoded ", Eweights);
				FinalModel.printWeights("b_"+layer+" After decoding ", weights);
				output.put(layer, weights);
				
			}
			
			
			output.put(layer, weights);	
			
		}
		
		
		return output;
		
	}


	
	private HashMap<String, double[][]> decodeWeights(HashMap<String, EncodedWeight[][]> encodedWeights)
	{
		
		HashMap<String, double[][]> decoded = new HashMap<String, double[][]>();
		
		Iterator<String> iterator = encodedWeights.keySet().iterator();
		
		while(iterator.hasNext())
		{
			String layer = iterator.next();
			
			EncodedWeight[][] eW = encodedWeights.get(layer);
			int[] shapeA = Matrix.getShape(eW);
			double[][] W= new double[shapeA[0]][shapeA[1]];
			for(int i =0; i< shapeA[0]; i++)
			{
				for(int j=0; j<shapeA[1]; j++)
				{
					double value = 0;
					Iterator<String> keys = eW[i][j].getKeys();
					while(keys.hasNext())
					{
						String hash = keys.next();
						double counter = eW[i][j].getItems(hash);
						//counter = (counter / (double) nodes);
						eW[i][j].replaceWeight(hash, counter);
						double aux =  decoder.decode(hash);
						aux = aux * counter;
						value = value + aux;
					}
					
					W[i][j] = value;
							
				}
		
			}
			encodedWeights.put(layer, eW);
			decoded.put(layer, W);
			
		}
		return decoded;
		
	}
	
	
	@Override
	public void run() {
		String name = this.loop+" Node"+this.node;
		long startTime = System.nanoTime();
		System.out.println(name+" started");

		History history;
		

			try 
			{
				history = mlp.fit(m, labels, this.mlp.getBatchSize());


				double[][] x_test = m;
				double[][] y_test = labels;

				//9:get the Metric data.
				Accuracy acc = history.getAccuracy(History.Type.TRAINING);
				Precision prec = history.getPrecision(History.Type.TRAINING);
				Recall rec = history.getRecall(History.Type.TRAINING);

				
				try
				{
					//10:show the training result
					String printing1 = Util.metricsTable(prec, rec, acc, y_test, this.activation);
					System.out.println(printing1);
					long endTime = System.nanoTime();
					long timeElapsed = endTime - startTime;

					//System.out.println("Execution time in nanoseconds: " + timeElapsed);
					System.out.println("Execution time in milliseconds: " + timeElapsed / 1000000);
					PlotTraining.start(name, history, printing1, this.window);

					
				}
				catch(NullPointerException e) {e.printStackTrace();}
				
				
				
				
//				//10:show the training result
//				String printing1 = Util.metricsTable(prec, rec, acc, y_test, this.activation);
//				System.out.println(printing1);
//				long endTime = System.nanoTime();
//				long timeElapsed = endTime - startTime;
//
//				//System.out.println("Execution time in nanoseconds: " + timeElapsed);
//				System.out.println("Execution time in milliseconds: " + timeElapsed / 1000000);
//				PlotTraining.start(name, history, printing1, this.window);

				HashMap<String, double[][]> weights = history.getAllWeights();
				HashMap<String, double[][]> bias = history.getAllBias();
				
				if(this.clip!=0)
				{
					weights = this.clip(weights, this.clip);
					bias = this.clip(bias, this.clip);
				}
					
				
				HashMap<String, EncodedWeight[][]> encodedWeights = this.encodeWeights(weights);
				HashMap<String, EncodedWeight[][]> encodedBias = this.encodeWeights(bias);
 
				Client client = new Client(this.server_ip, this.server_port);
				
				ArrayList<HashMap<String, EncodedWeight[][]>> woi=new ArrayList<HashMap<String, EncodedWeight[][]>>();
				
				woi.add(encodedWeights);
				if(this.isBias)
					woi.add(encodedBias);
				
				ArrayList<HashMap<String, EncodedWeight[][]>> finalweights = client.sendEncodedWeights(woi);
				
				encodedWeights=null;
				encodedBias=null;
				HashMap<String, double[][]> modelWeights = null;
				HashMap<String, double[][]> modelBias = null;
				
				if(finalweights.size()>=1)
				{
					encodedWeights = finalweights.get(0);
					modelWeights = this.finalizeEncWeights(encodedWeights, false);
					if(finalweights.size()>1)
					{
						encodedBias = finalweights.get(1);
						modelBias = this.finalizeEncWeights(encodedBias, true);
					}
						
				}
				
				synchronized (aggregator) 
				{

					
					aggregator.setFinalWeights(modelWeights);
					aggregator.setFinalEncodedWeights(encodedWeights);
					
					
					if(encodedBias!=null)
					{
						aggregator.setFinalBias(modelBias);
						aggregator.setFinalEncodedBias(encodedBias);
					}
						
					

					aggregator.decreaseCounter();
					aggregator.notifyAll();
					// msg.notifyAll();
					
					
				}
                
				
				

			

			} catch (MatrixException | IOException | ClassNotFoundException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			System.out.println("Thread "+name+" dies!!!");
		
	}

	
	
	private double checkRange(double w)
	{
		if(w<this.min)
			return this.min;
		else if(w > this.max)
			return this.max;
		else return w;
		
	}
	
	
	private HashMap<String, EncodedWeight[][]> encodeWeights(HashMap<String, double[][]> weights)
	{
		HashMap<String, EncodedWeight[][]> encoded = new HashMap<String, EncodedWeight[][]>();
		
		Iterator<String> iterator = weights.keySet().iterator();
		
		while(iterator.hasNext())
		{
			String layer = iterator.next();
			double[][] W = weights.get(layer);
			int[] shapeA = Matrix.getShape(W);
			EncodedWeight[][] eW= new EncodedWeight[shapeA[0]][shapeA[1]];
			for(int i =0; i< shapeA[0]; i++)
			{
				for(int j=0; j<shapeA[1]; j++)
				{
					double nW = this.checkRange(W[i][j]);
					Vector<String> hc = encoder.encode(nW);
					String aux = encoder.getLastChannelValue(hc);
					eW[i][j] = new EncodedWeight();
					eW[i][j].addWeight(aux);
				}
		
			}
			
			encoded.put(layer, eW);
			
		}
		return encoded;
		
	}
	

}
