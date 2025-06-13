package f21bc.labs.Federated.threads.scenario;

import java.io.File;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.Exceptions.ConfigurationException;
import f21bc.labs.Federated.ModelAggregator;
import f21bc.labs.Federated.Utils;
import f21bc.labs.Federated.Noise.NoiseGenerator;
import f21bc.labs.Federated.hashComb.HCDecoder;
import f21bc.labs.Federated.hashComb.HCEncoder;
import f21bc.labs.Federated.threads.EncodedNodeThread;
import f21bc.labs.Federated.threads.NodeThread;
import f21bc.labs.Federated.utils.EncodedWeight;
import f21bc.labs.LF.LossFunction;
import f21bc.labs.LF.LossFunction.Types;
import f21bc.labs.NN.EarlyStopping;


public class ClientManager2 {
	
	public static void main(String[] args) throws ClassNotFoundException, IOException
	{
		Nd4j.setDataType(DataType.DOUBLE);
		
		String subDir = "HFL";
		String dir = "data"+File.separator+subDir+File.separator;
		
		
		String propFile = "configuration.prop";
		Properties prop = Utils.loadConf(propFile);
		NoiseGenerator noiser = null;
		
		String file = Utils.getDataFile(prop);
		
		boolean bias = Utils.includesBias(prop);
		
		String encFile = Utils.getEncodingFile(prop);
		
		
		boolean hash = Utils.isHashing(prop);
		boolean windows=Utils.allow4Window(prop); 
		
		if(!hash)
			noiser = Utils.getNoiseGenerator(prop);
		
		int channels= Utils.getChannels(prop);
		double min = Utils.getMin(prop);
		double max = Utils.getMax(prop);
		double clip = Utils.getClip(prop);
		int nodes = Utils.getNodes(prop);
		int epochs = Utils.getEpochs(prop);
		int iterations = Utils.getIterations(prop);
		int[] layers= Utils.getLayers(prop);
		double learning_rate = Utils.getLR(prop);
		double batchSize = Utils.getBatchSize(prop);
		double clipping = Utils.getGclipping(prop);
		Types loss_function = LossFunction.Types.CE;
		
		ActivationFunction.Types af = Utils.getActivationFunc(prop);
		
		String modelClass = Utils.getModelClass(prop); 
		int classes = Utils.getClasses(prop);
		
		modelClass = modelClass+"/"+classes;
		
//		double learning_rate = 0.06;
		
//		double learning_rate = -1;
//		double learning_rate = 0.005;
		
		ModelAggregator aggregator = new ModelAggregator(hash);
		
		HCEncoder encoder = null;
		HCDecoder decoder = null;
		if(hash)
		{
			encoder = new HCEncoder(channels, min, max, hash);
			encoder.store(encFile);
			decoder = new HCDecoder(encFile);
		}
		
		
		Path output = Paths.get("TOTAL.txt");
		Path nodeJ = null;
		
		
		
		nodeJ = Paths.get("TEMP");
		String[]entries = nodeJ.toFile().list();
		for(String s: entries){
		    File currentFile = new File(nodeJ.toString(),s);
		    currentFile.delete();
		}
		
		
		try {
            Files.deleteIfExists(output);
            System.out.println("Deleted file: "+output);
        }
        catch (NoSuchFileException e) {
            System.out.println(
                "No such file/directory exists");
        }
		
		
		
		
		File folder = new File(dir);
		File[] listOfFiles = folder.listFiles();
		
		List<File> datas
        = new ArrayList<File>();

		 // Output file
	    
	 // Charset for read and write
	    Charset charset = StandardCharsets.UTF_8;
	    List<String> lines = null;
	    
		for (File f : listOfFiles) {
		    if (f.isFile()) {
		        
//		        datas.add(f);
		    	lines = Files.readAllLines(f.toPath(), charset);
		    	lines.remove(0);
		    	Files.write(output, lines, charset, StandardOpenOption.CREATE,
		                StandardOpenOption.APPEND);
		    }
		}
		lines = Files.readAllLines(output, charset);
		System.out.println("Total size: "+lines.size());
		int rows = lines.size()/nodes; 
		System.out.println("Size per file: "+rows);
		
		
		
		
//		System.exit(1);

		int indexS = 0;
		int indexE = 0;
		
		
		for(int j=0; j<nodes; j++)
		{
			
			indexE = indexS+rows;
			if(j==(nodes-1))
				indexE = (lines.size()-1);
			
			System.out.println("start: "+indexS+"   End: "+indexE);
			
			nodeJ = Paths.get("TEMP"+File.separatorChar+"Node"+j+".csv");
			
			List<String> newL =  lines.subList(indexS, indexE);
			Files.write(nodeJ, newL, charset, StandardOpenOption.CREATE,
	                StandardOpenOption.APPEND);
			if(indexE<lines.size()-1)
				indexS = indexE+1;
			else indexS = lines.size()-1;
		}
		
		
		for(int i=0; i<iterations; i++)
		{
			//aggregator.setCounter(datas.size());
			aggregator.setCounter(nodes);
			
			
				//6:determine the early stop
//				EarlyStopping earlyStop = new EarlyStopping(0.00001, 10);
//				EarlyStopping earlyStop = new EarlyStopping(0.00001, 20);
				EarlyStopping earlyStop = new EarlyStopping(0.000001, 60);				
				
				
//				String path = dir+file+j+".csv";
				for(int j=0; j<nodes; j++)
				{
					nodeJ = Paths.get("TEMP"+File.separatorChar+"Node"+j+".csv");
					String path = nodeJ.toString();
					System.out.println("adding data file: "+path);
				
				try 
				{
					NodeThread thread;
					if(!hash)
					{
						thread = new NodeThread(i, j, path, epochs, learning_rate, layers, af, loss_function, earlyStop, bias, modelClass, batchSize, clipping);
						if(noiser!=null)
							thread.setnoiseGen(noiser);
					}
					
					else
					{
						thread = new EncodedNodeThread(i, j, path, epochs, learning_rate, layers, af, loss_function, earlyStop, bias, modelClass, batchSize, clipping);
						((EncodedNodeThread) thread).configure(encoder);
						
					}
					
					
					if(clip!=0)
						thread.setClipping(clip);
					
					thread.setServer(Utils.getGMServer_IP(prop), Utils.getGMServer_Port(prop));
					thread.allow4Windows(windows);
					thread.setAggregator(aggregator);

					//at first iteration the weights are set randomly
					if(i>0)
						thread.initializeNN(bias);

					
					else
					{
						File initFile = new File(Utils.getInitWeightsFile(prop));
						if(initFile.exists() && !initFile.isDirectory()) 
						{ 
						    
							FileInputStream f = new FileInputStream(initFile);
				            ObjectInputStream in = new ObjectInputStream(f);
				            
				           
				            ArrayList woi=new ArrayList<>();
				            woi=(ArrayList)in.readObject();

				            HashMap<String, double[][]> weights=null;
				            HashMap<String, EncodedWeight[][]> encWeights=null;
				            HashMap<String, double[][]> b=null;
				            HashMap<String, EncodedWeight[][]> encB=null;
				            
				            if(woi.size()>0)
				            {
				            	if(!hash)
				            	{
				            		weights = (HashMap<String, double[][]>)woi.get(0);
				            		//weights = thread.finalizeWeights(weights, false);
				            		thread.printWeights(weights, false);
					            	// Method for deserialization of object
						            System.out.println("Initializing weights from "+initFile.getName()+" Size: "+woi.size());
						            
						            if(woi.size()>1)
						            {
						            	b = (HashMap<String, double[][]>)woi.get(1);
						            	//b = thread.finalizeWeights(b, true);
						            	thread.printWeights(b, true);
						            	thread.initializeNN(weights, b);
		
						            }
						            else 
						            	thread.initializeNN(weights);
				            	}
				            	
				            	else
				            	{
				            		encWeights = (HashMap<String, EncodedWeight[][]>)woi.get(0);
					            	// Method for deserialization of object
				            		
				            		System.out.println("Initializing weights from "+initFile.getName()+" Size: "+woi.size());
						            
						            if(woi.size()>1)
						            {
						            	encB = (HashMap<String, EncodedWeight[][]>)woi.get(1);
						            	thread.initializeNN(encWeights, encB);
						            	
						            }
						            else 
						            	thread.initializeNN(encWeights);
				            	}
				            	
				            	
				            }
				             
				            
				            
							
						}
						
					}
					
					
					thread.start();
					
					
					
				
				} catch (IllegalStateException | FileNotFoundException | ConfigurationException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
			
		
			
			synchronized (aggregator) {
	            try
	            {
	            	
	                System.out.println("Main Thread waiting to get notified at time:"+System.currentTimeMillis()+" iteration "+i);
	                while(!aggregator.isFinilazed())
	                aggregator.wait();
	                System.out.println("Main Thread got notified at time:"+System.currentTimeMillis()+" iteration "+i);
	                System.out.println("\n\nClient End of cycle "+i);
	            }catch(InterruptedException e){
	                e.printStackTrace();
	            }
			}
			
		}
		
		
		System.out.println("\n\nCLIENT Never reach this point?");
		

	}

}
