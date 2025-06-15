package ebtic.labs.Federated.threads;

import java.io.FileOutputStream;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import ebtic.labs.Federated.ModelAggregator;
import ebtic.labs.Federated.Utils;
import ebtic.labs.Federated.hashComb.HCDecoder;
import ebtic.labs.Federated.hashComb.HCEncoder;
import ebtic.labs.Federated.utils.EncodedWeight;
import ebtic.labs.utils.CoinToss;
import ebtic.labs.utils.Matrix;
import ebtic.labs.utils.CoinToss.Coin;
import khalifa.ebtic.security.hashcomb.tree.Tree;

public class EncodedGeneralModelThread extends GeneralModelThread{

	
	protected boolean randomClient = false;
	 
	
	public EncodedGeneralModelThread(ModelAggregator aggregator, int nodes) {
		super(aggregator, nodes);
		// TODO Auto-generated constructor stub
		if(this.configuration.containsKey("encoding.random.clients"))
		{
			this.randomClient = Utils.includesRandomClients(this.configuration);
//			System.out.println("IS IT RANDOM: "+randomClient);
			if(this.randomClient)
				aggregator.setRandomness();
		}
			
	}


	
	@Override
    public void run() {
        String name = Thread.currentThread().getName();
        
        try 
        {
			server.receiveweights(true);
		} catch (ClassNotFoundException | IOException | InterruptedException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
         
        
        synchronized (this.aggregator) 
        {
            try
            {
                System.out.println(name+" waiting to get notified at time:"+System.currentTimeMillis());
                
                while(!this.aggregator.isFinilazed())
                	this.aggregator.wait();
                
                System.out.println(name+" waiter thread got notified at time:"+System.currentTimeMillis());
                
                HashMap<String, EncodedWeight[][]> encodedWeights = this.aggregator.getWeights();
                HashMap<String, EncodedWeight[][]> encodedBias = this.aggregator.getBias();
                
                //here finalize the weights by calculating the average... 
                encodedWeights = this.finalizeEncodedWeights(encodedWeights);
                encodedBias = this.finalizeEncodedWeights(encodedBias);
                
                
                
                ArrayList<HashMap<String, EncodedWeight[][]>> allWeights = new ArrayList<HashMap<String, EncodedWeight[][]>>();
                
                allWeights.add(encodedWeights);
                if(encodedBias!=null)
                	allWeights.add(encodedBias);
                
                /// need to include also the bias
                this.server.sendUpdatedEncodedWeights(allWeights);
                if(this.lastIter)
                	Utils.dumpEnc2File(allWeights, "ModelEnc.ser");
                		
                
                aggregator.isCompleted=true;
    			aggregator.notifyAll();
    			// msg.notifyAll();
                
            }catch(InterruptedException | IOException e){
                e.printStackTrace();
            }
            
        }
    }

	
	
	private int randomizedClients(EncodedWeight w)
	{
		
		Iterator<String> keys = w.getKeys();
		int clients= 0;
		int randoms=0;
		
		CoinToss toss = new CoinToss();
		while(keys.hasNext())
		{
			String hash = keys.next();
			int counter =(int) w.getItems(hash);
			clients = clients + counter; 
			randoms = randoms + counter; 
			int aux = counter;
			if(counter>0)
			{
				for(int i = 1; i<= counter; i++)
					if(toss.flip()==CoinToss.Coin.Heads)
					{
						aux = aux -1;
						randoms = randoms - 1;
					}
			}
			
//			if(aux<=0)
//				w.removeWeight(hash);
//			else
			w.replaceWeight(hash, aux);
		}
		
		return randoms;
		
	}
	
	
	
	public HashMap<String, EncodedWeight[][]> finalizeEncodedWeights(HashMap<String, EncodedWeight[][]> encodedWeights)
	{
		
		Iterator<String> iterator = encodedWeights.keySet().iterator();
		
		while(iterator.hasNext())
		{
			String layer = iterator.next();
			
			EncodedWeight[][] eW = encodedWeights.get(layer);
			int[] shapeA = Matrix.getShape(eW);
			//double[][] W= new double[shapeA[0]][shapeA[1]];
			
			
			int finalNodes = this.nodes;
			
			if(this.randomClient)
				finalNodes = this.aggregator.getRandomNodes(); 
//				randomNodes = randomizedClients(eW[i][j]);
			System.out.println("Total Node: "+this.nodes+"  //  randomized: "+finalNodes);
			
			for(int i =0; i< shapeA[0]; i++)
			{
				for(int j=0; j<shapeA[1]; j++)
				{
					double value = 0;
					

					Iterator<String> keys = eW[i][j].getKeys();
					
					
					double total = 0;
					while(keys.hasNext())
					{
						String hash = keys.next();
						double counter = eW[i][j].getItems(hash);
						total = total + counter;
						if(counter>0)
							counter = (counter / (double) finalNodes);
						eW[i][j].replaceWeight(hash, counter);
						
					}
//					System.out.println("Counted nodes: "+total);
							
				}
		
			}
			encodedWeights.put(layer, eW);
			
			
		}
		return encodedWeights;
		
	}
	
	
}
