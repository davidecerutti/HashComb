package f21bc.labs.Federated.threads;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Properties;

import f21bc.labs.Federated.ModelAggregator;
import f21bc.labs.Federated.Utils;
import f21bc.labs.Federated.communications.Server;
import f21bc.labs.utils.Matrix;

public class GeneralModelThread extends Thread {

	protected boolean randomClient = false;
	protected ModelAggregator aggregator;
	protected int nodes;
	protected Properties configuration;
	protected Server server;
	protected boolean lastIter = false;
	
	public GeneralModelThread(ModelAggregator aggregator, int nodes)
	{
		this.aggregator = aggregator;
		this.nodes = nodes;
		
		String propFile = "configuration.prop";
		this.configuration = Utils.loadConf(propFile);
		if(this.configuration.containsKey("encoding.random.clients"))
		{
			this.randomClient = Boolean.parseBoolean((String) this.configuration.get("encoding.random.clients"));
			if(this.randomClient)
				aggregator.setRandomness();
		}
		
	}
	
	
	
	public void set_last_iteration()
	{
		this.lastIter = true;
	}

	

	public void setCommunicationServer(Server server)
	{
		this.server = server;
	}
	
	
	@Override
    public void run() {
        String name = Thread.currentThread().getName();
        
        
        try 
        {
			server.receiveweights(false);
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
                
                HashMap<String, double[][]> modelWeights = this.aggregator.getWeights();
                HashMap<String, double[][]> modelBias = this.aggregator.getBias();
                
                
                
                //here finalize the weights by calculating the average... 
                modelWeights = this.finalizeWeights(modelWeights, false);
                modelBias = this.finalizeWeights(modelBias, true);
                
                ArrayList<HashMap<String, double[][]>> allWeights = new ArrayList<HashMap<String, double[][]>>();
                
                allWeights.add(modelWeights);
                if(modelBias!=null)
                	allWeights.add(modelBias);
                
                
                this.server.sendUpdatedWeights(allWeights);
                if(this.lastIter)
                	Utils.dump2File(allWeights, "Model.ser");
                
              
                
                aggregator.isCompleted=true;
    			aggregator.notifyAll();
    			// msg.notifyAll();
                
            }catch(InterruptedException | IOException e){
                e.printStackTrace();
            }
            
        }
    }
	
	public HashMap<String, double[][]> finalizeWeights(HashMap<String, double[][]> modelWeights, boolean bias)
	{
		HashMap<String, double[][]> output = new HashMap<String, double[][]>();
		Iterator iter = modelWeights.keySet().iterator();
		String type = "W_";
		if(bias)
			type="b_";
		while(iter.hasNext())
		{
			String layer = (String) iter.next();
			double[][] weights = modelWeights.get(layer);
			FinalModel.printWeights(type+layer+" Before dividing by "+this.aggregator.getNodes(), weights);
			
			int finalNodes = this.aggregator.getNodes();
			
			if(this.randomClient)
				finalNodes = this.aggregator.getRandomNodes(); 
//			System.out.println("Total Node: "+this.nodes+"  //  randomized: "+finalNodes);
			weights = Matrix.divide(weights, finalNodes);
			FinalModel.printWeights(type+layer+" After dividing by "+finalNodes, weights);
			output.put(layer, weights);	
			
		}
	
		return output;
		
	}
	
}
