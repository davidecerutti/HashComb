package f21bc.labs.Federated.threads;

import java.io.FileNotFoundException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.AF.ActivationFunction.Types;
import f21bc.labs.Exceptions.ConfigurationException;
import f21bc.labs.Exceptions.MatrixException;
import f21bc.labs.Federated.ModelAggregator;
import f21bc.labs.Federated.Noise.NoiseGenerator;
import f21bc.labs.Federated.communications.Client;
import f21bc.labs.Federated.utils.EncodedWeight;
import f21bc.labs.LF.LossFunction;
import f21bc.labs.NN.EarlyStopping;
import f21bc.labs.NN.Layer;
import f21bc.labs.NN.MLModelFactory;

import f21bc.labs.NN.NN;
import f21bc.labs.NN.dl4j.utils.Utils;
import f21bc.labs.metrics.Accuracy;
import f21bc.labs.metrics.History;
import f21bc.labs.metrics.Precision;
import f21bc.labs.metrics.Recall;
import f21bc.labs.objects.Generic;
import f21bc.labs.objects.MyObject;
import f21bc.labs.utils.Matrix;
import f21bc.labs.utils.ObjectSizeFetcher;
import f21bc.labs.utils.PlotTraining;
import f21bc.labs.utils.Util;
import f21bc.labs.utils.WeightsExtractor;

public class NodeThread extends Thread {

	protected NN mlp;
	
	protected ModelAggregator aggregator;
	protected String inputString;
	protected double[][] m;
    protected double[][] labels;
    protected Types activation;
    protected int loop;
    protected int node;
    protected String server_ip= "localhost";
    protected int server_port = 4444;
    protected boolean isBias;
    protected NoiseGenerator noiser = null;
    protected boolean window = true;
    protected double clip = 0;
    protected double grad_clipping=1;
    
	
	public NodeThread(int iteration, int node, String filenameString, int epochs, double learning_rate, int[] layers, ActivationFunction.Types activation_function, LossFunction.Types loss_function, EarlyStopping ea, boolean bias, String modelClass, double batchSize, double clipping) throws ConfigurationException, IllegalStateException, FileNotFoundException
	{
		this.loop = iteration;
		this.node = node;
		this.inputString = filenameString;
		List<MyObject> beans = Generic.instanciateCSV(filenameString);
		this.isBias=bias;
		//beans.forEach(System.out::println);
		System.out.println("SIZE: "+beans.size());
		//2:Shuffle the dataset
		Collections.shuffle(beans);
		//3:get the train data and labels
		this.m = Matrix.getData(beans);
		this.labels = Matrix.getLabels(beans);	
		this.activation = activation_function;
		this.grad_clipping = clipping;
		String[] aux = modelClass.split("/");
		
		modelClass = aux[0];
		int classes = Integer.parseInt(aux[1]);
		
		this.mlp = MLModelFactory.getModelType(modelClass, epochs, learning_rate, layers, activation_function, loss_function ,ea, classes, clipping); 
//		this.mlp = new MLP(epochs, learning_rate, layers, activation_function, loss_function ,ea);
//		this.mlp.setClasses(classes);
		try {
			this.mlp.setBatchSize(batchSize);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public void saveModel(String directory) throws IOException
	{
		this.mlp.save(directory);
	}
	
	
	public void allow4Windows(boolean allow)
	{
		this.window = allow;
	}
	
	public void setnoiseGen(NoiseGenerator generator)
	{
		this.noiser = generator;
	}
	
	
	public void setServer(String ip, int port)
	{
		this.server_ip = ip;
		this.server_port = port;
	}
	
	
	public void setAggregator(ModelAggregator aggregator)
	{
		this.aggregator = aggregator;
	}
	
	
	public void setClipping(double clip)
	{
		this.clip = clip;
	}
	
	public void initializeNN(boolean bias)
	{
		//HashMap weights = this.aggregator.getWeights();
		HashMap<String, double[][]> weights = this.aggregator.getFinalWeights();
		if(!bias)
			mlp.instanciate(weights);
		else
		{
			HashMap<String, double[][]> b = this.aggregator.getFinalBias();
			
			System.out.println(ModelAggregator.printWeightSize(weights, false));
			System.out.println(ModelAggregator.printWeightSize(b, true));
						
			mlp.instanciate(weights, b);
		}
		
		
	}
	
	public void initializeNN(HashMap weights)
	{
		//HashMap weights = this.aggregator.getWeights();
		mlp.instanciate(weights);
		
		
	}
	
	
	public void initializeNN(HashMap weights, HashMap bias)
	{
		//HashMap weights = this.aggregator.getWeights();
		mlp.instanciate(weights, bias);
	}
	

	
	
	public void printWeights(HashMap<String, double[][]> modelWeights, boolean bias)
	{
		Iterator iter = modelWeights.keySet().iterator();
		while(iter.hasNext())
		{
			String layer = (String) iter.next();
			double[][] weights = modelWeights.get(layer);
			
			if(!bias)
			{
				FinalModel.printWeights("W_"+layer+" Received ", weights);
		
			}
			else
			{
				FinalModel.printWeights("b_"+layer+" Received ", weights);
				
			}
			
		}
	
	}
	
	
	
	private HashMap<String, double[][]> addNoise(HashMap<String, double[][]> weights)
	{
		double mean = 0;
		
		Iterator iter = weights.keySet().iterator();
		while(iter.hasNext())
		{
			
			String L = (String) iter.next();
			
			double[][] W = weights.get(L);
			
			W = this.noiser.addNoise(W);
			
			weights.put(L, W);
		}
	
		
		return weights;
		
	}

	
	protected HashMap<String, double[][]> clip(HashMap<String, double[][]> weights, double clip)
	{
		
		Iterator iter = weights.keySet().iterator();
		while(iter.hasNext())
		{
			
			String L = (String) iter.next();
			
			double[][] W = weights.get(L);
			
			W = Matrix.clip(W, clip);
			
			weights.put(L, W);
		}
	
		
		return weights;
		
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
				
				
				HashMap<String, double[][]> weights = history.getAllWeights();
				HashMap<String, double[][]> bias = history.getAllBias();
				
//				Utils.printSize(bias);
				
				
				if(this.clip!=0)
				{
					weights = this.clip(weights, this.clip);
					bias = this.clip(bias, this.clip);
				}
					
				
				
				
				// insert the NOISE HERE:
				
				if(this.noiser!=null)
				{
					
					this.noiser.setVariance(this.mlp.getEpochs(), Matrix.getShape(this.m)[0], this.mlp.getLearningRate(), this.mlp.getBatchSize(), this.grad_clipping);
					this.noiser.setMean(0);
					
					weights = this.addNoise(weights);
					bias = this.addNoise(bias);
				}
				
				
				////////////////////////
				
				
				
				
				
				ArrayList<HashMap<String, double[][]>> woi=new ArrayList<HashMap<String, double[][]>>();
				
				woi.add(weights);
				if(this.isBias)
					woi.add(bias);
				
		
				Client client = new Client(this.server_ip, this.server_port);
				
//				System.out.println("MESSAGE SIZE: "+ObjectSizeFetcher.getObjectSize(woi));
				
				ArrayList<HashMap<String, double[][]>> finalweights = client.sendWeights(woi);
				
				
				weights=null;
				bias=null;
				HashMap<String, double[][]> modelWeights = null;
				HashMap<String, double[][]> modelBias = null;
				
				if(finalweights.size()>=1)
				{
					modelWeights = finalweights.get(0);
					this.printWeights(modelWeights, false);
					//modelWeights = this.finalizeWeights(weights, false);
					
					if(finalweights.size()>1)
					{
						modelBias = finalweights.get(1);
						//modelBias = this.finalizeWeights(bias, true);
						this.printWeights(modelBias, true);
					}
						
				}
				
				
				synchronized (this.aggregator) 
				{

					
					aggregator.setFinalWeights(modelWeights);
					if(modelBias!=null)
						aggregator.setFinalBias(modelBias);
					aggregator.decreaseCounter();
					aggregator.notifyAll();
				}	
					
				
					
			} catch (MatrixException | IOException | ClassNotFoundException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			
			try {
				this.saveModel("data/tests/N4JModels");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			System.out.println("Thread "+name+" dies!!!");
		

	}
	
	

}
