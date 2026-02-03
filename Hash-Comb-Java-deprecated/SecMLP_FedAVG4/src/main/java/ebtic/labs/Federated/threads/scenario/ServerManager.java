package ebtic.labs.Federated.threads.scenario;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.Exceptions.ConfigurationException;
import ebtic.labs.Federated.ModelAggregator;
import ebtic.labs.Federated.Utils;
import ebtic.labs.Federated.communications.Server;
import ebtic.labs.Federated.hashComb.HCDecoder;
import ebtic.labs.Federated.hashComb.HCEncoder;
import ebtic.labs.Federated.threads.EncodedGeneralModelThread;
import ebtic.labs.Federated.threads.GeneralModelThread;
import ebtic.labs.Federated.utils.EncodedWeight;
import ebtic.labs.LF.LossFunction;
import ebtic.labs.NN.EarlyStopping;

public class ServerManager {
	
	
	
	
	
	private static boolean finalize_weights(GeneralModelThread modelThread, ModelAggregator aggregator, boolean hash) throws ClassNotFoundException, IOException
	{
		
			HashMap weights = aggregator.getWeights();
            HashMap bias = aggregator.getBias();
            
          
            
            ArrayList allWeights = new ArrayList<>();
            
            allWeights.add(weights);
            allWeights.add(bias);
            System.out.println("Finalizing weights!!!!!!!!!!");
            return initializeAggregator(modelThread, allWeights, aggregator, hash);
	}
	
	
	
	
	private static boolean init_from_file(GeneralModelThread modelThread, Properties prop, ModelAggregator aggregator, boolean hash) throws IOException, ClassNotFoundException
	{
		File initFile = new File(Utils.getInitWeightsFile(prop));
		ArrayList woi = null;
		
		boolean init_from = true;
		
		if(initFile.exists() && !initFile.isDirectory() && init_from) 
		{
			FileInputStream f = new FileInputStream(initFile);
            ObjectInputStream in = new ObjectInputStream(f);
            
            woi=new ArrayList<>();
            woi=(ArrayList)in.readObject();
            System.out.println("Initializing weights from "+initFile.getName());
            return initializeAggregator(modelThread, woi, aggregator, hash);
            
		}
		
		return false;
		
		
	}
	
	
	
	
	private static boolean initializeAggregator(GeneralModelThread modelThread, ArrayList woi, ModelAggregator aggregator, boolean hash) throws IOException, ClassNotFoundException
	{
		
            HashMap<String, double[][]> weights=null;
            HashMap<String, EncodedWeight[][]> encWeights=null;
            HashMap<String, double[][]> b=null;
            HashMap<String, EncodedWeight[][]> encB=null;
            
            if(woi.size()>0)
            {
            	if(!hash)
            	{
            		weights = (HashMap<String, double[][]>)woi.get(0);
            		//mauri//weights = modelThread.finalizeWeights(weights);
	            	// Method for deserialization of object
		            
		            
		            if(woi.size()>1)
		            {
		            	b = (HashMap<String, double[][]>)woi.get(1);
		            	//mauri//b = modelThread.finalizeWeights(b);
		            	//aggregator.setWeights(weights);
		            	//aggregator.setBias(b);
		            	aggregator.setFinalWeights(weights);
		            	aggregator.setFinalBias(b);
		            	
		            	
		            }
		            else 
		            	//aggregator.setWeights(weights);
		            	aggregator.setFinalWeights(weights);
            	}
            	
            	else
            	{
            		encWeights = (HashMap<String, EncodedWeight[][]>)woi.get(0);
            		//mauri//encWeights = ((EncodedGeneralModelThread) modelThread).finalizeEncodedWeights(encWeights);
	            	// Method for deserialization of object
		            
		            
		            if(woi.size()>1)
		            {
		            	encB = (HashMap<String, EncodedWeight[][]>)woi.get(1);
		            	//mauri//encB = ((EncodedGeneralModelThread) modelThread).finalizeEncodedWeights(encB);
		            	
		            	
		            	aggregator.setFinalEncodedWeights(encWeights);
		            	aggregator.setFinalEncodedBias(encB);
		            	
		            }
		            else 
		            	//aggregator.setEncodedWeights(encWeights);
		            	aggregator.setFinalEncodedWeights(encWeights);
            	}
            	
            	return true;
            }
             
        
		return false;
		
	}
	
	
	
	public static void main(String[] args) throws IOException
	{
		
//		String subDir = "HFL";
//		String dir = "data"+File.separator+subDir+File.separator;
//		String file = "spam_hf_P";
		
		Nd4j.setDataType(DataType.DOUBLE);
		
		String propFile = "configuration.prop";
		Properties prop = Utils.loadConf(propFile);
		System.out.println("Starting a new Server: "+prop.hashCode());
		
		String encFile = Utils.getEncodingFile(prop);
		
		boolean hash = Utils.isHashing(prop);
		int channels= Utils.getChannels(prop);
		double min = Utils.getMin(prop);
		double max = Utils.getMax(prop);
		
		int nodes = Utils.getNodes(prop);
//		int epochs = Utils.getEpochs(prop);
		int iterations = Utils.getIterations(prop);
//		int[] layers= Utils.getLayers(prop);
		
		
		ActivationFunction.Types af = Utils.getActivationFunc(prop);
		
		//6:determine the early stop
		EarlyStopping earlyStop = new EarlyStopping(0.0015, 20);
//		EarlyStopping earlyStop = new EarlyStopping(0.00015, 20);
		
		double learning_rate = 0.05;
		
		ModelAggregator aggregator = new ModelAggregator(hash);
		
		Server server = new Server(Utils.getGMServer_Port(prop), aggregator);
		
		HCEncoder encoder = null;
		
		if(hash)
		{
			encoder = new HCEncoder(channels, min, max, hash);
			encoder.store(encFile);
			
		}
		for(int i=0; i<iterations; i++)
		{
			//ExecutorService threadPool = Executors.newFixedThreadPool(nodes+1);
			//ExecutorService threadPool = Executors.newFixedThreadPool(1);
			
			server.resetConnections();
			
			
			
			System.out.println("Iteration number "+(i+1));
			
			aggregator.setCounter(nodes);
			
			
			GeneralModelThread modelThread;
			if(!hash)
				modelThread = new GeneralModelThread(aggregator, nodes);
			else
			{
				modelThread = new EncodedGeneralModelThread(aggregator, nodes);
				
			}

			if(i==(iterations-1))
			
				modelThread.set_last_iteration();
				
			try 
			{
				
				if(i>0)
					System.out.println("Exit: "+finalize_weights(modelThread, aggregator, hash));
				else
					init_from_file(modelThread, prop, aggregator, hash);

			} catch (ClassNotFoundException | IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}


			
			modelThread.setCommunicationServer(server);
			modelThread.start();
			
			synchronized (aggregator) {
	            try
	            {
	            	
	                System.out.println("Main Thread waiting to get notified at time:"+System.currentTimeMillis()+" iteration "+i);
	                while(!aggregator.isFinilazed())
	                aggregator.wait();
	                System.out.println("\n\nServer End of cycle "+i);
	            }catch(InterruptedException e){
	                e.printStackTrace();
	            }
			}
			
		}
		System.out.println("\n\nServer Never reach this point?");
		server.closeConnection();
	}

}
