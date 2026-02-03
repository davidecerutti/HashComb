package ebtic.labs.Federated.communications;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import ebtic.labs.Federated.ModelAggregator;
import ebtic.labs.Federated.utils.EncodedWeight;

public class Server {
	
	private int port;
	private ModelAggregator aggregator;
	
    public static final String GREETING = "Hello I must be going.\r\n";
    private ServerSocketChannel ssc = null;
    private List<ObjectOutputStream> channelsL = new ArrayList<ObjectOutputStream>();
    
    public Server(int port, ModelAggregator aggregator) throws IOException
    {
    	
    	this.ssc = ServerSocketChannel.open();
        this.ssc.socket().bind(new InetSocketAddress(port));
        this.ssc.configureBlocking(true);
    	
		this.port = port;
		this.aggregator = aggregator;
		
    }
    
    
    public void closeConnection() throws IOException
    {
    	this.ssc.socket().close();
    }
    
    public void resetConnections()
    {
    	channelsL = new ArrayList<ObjectOutputStream>();
    }
    
    
    public void receiveweights(boolean encoded) throws IOException, ClassNotFoundException, InterruptedException{
    
    	 

        while (!aggregator.isFinilazed()) {
          System.out.println("Waiting for connections");

          SocketChannel sc = ssc.accept();

          if (sc == null) {
            Thread.sleep(2000);
          } else {
            System.out.println("Incoming connection from: " + sc.socket().getRemoteSocketAddress());
            
            ObjectInputStream is = new ObjectInputStream(sc.socket().getInputStream());
            ObjectOutputStream os = new ObjectOutputStream(sc.socket().getOutputStream());

            
            
            if(!encoded)
            {
            	ArrayList<HashMap<String, double[][]>> message =(ArrayList<HashMap<String, double[][]>>)is.readObject();
            	this.processWeights(message);
            }
            	
            else
            {
            	ArrayList<HashMap<String, EncodedWeight[][]>> message =(ArrayList<HashMap<String, EncodedWeight[][]>>)is.readObject();
            	this.processEncodedWeights(message);
            }
            	
            
            System.out.println("Remaining Clients "+aggregator.counter);
            
            channelsL.add(os);
            
            
          }
        }
        
        
        
    }
    
    
    
    public void sendUpdatedWeights(ArrayList<HashMap<String, double[][]>> modelAllWeights) throws IOException
    {
    	Iterator<ObjectOutputStream> iter = this.channelsL.iterator();
    	int counter = 1;
    	while(iter.hasNext())
    	{
    		ObjectOutputStream os = iter.next();
    		System.out.println("Sending weights to client_"+counter);
    		os.writeObject(modelAllWeights);
    		os.flush();
    		os.close();
    		counter++;
    	}
    }

    
    public void sendUpdatedEncodedWeights(ArrayList<HashMap<String, EncodedWeight[][]>> encodedAllWeights) throws IOException
    {
    	synchronized (this.aggregator) 
    	{
    		Iterator<ObjectOutputStream> iter = this.channelsL.iterator();
    		int counter = 1;
    		while(iter.hasNext())
    		{
    			ObjectOutputStream os = iter.next();
    			System.out.println("Sending weights to client_"+counter);
    			os.writeObject(encodedAllWeights);
    			os.flush();
    			os.close();
    			counter++;
    		}
    	}
    }


    private void processWeights(ArrayList<HashMap<String, double[][]>> message) {
        
    	HashMap<String, double[][]> weights = null;
    	HashMap<String, double[][]> bias = null;
    	
    	if(message.size()>0)
        {
        	weights = (HashMap<String, double[][]>)message.get(0);
        	
        	boolean shall_we_remove = aggregator.removeWeight();
        	
        	aggregator.setWeights(weights, shall_we_remove);
            if(message.size()>1)
            {
            	bias = (HashMap<String, double[][]>)message.get(1);
            	aggregator.setBias(bias, shall_we_remove);
            }
            
        }
    	
    
    	aggregator.decreaseCounter();
    	
    }
    
    
    private void processEncodedWeights(ArrayList<HashMap<String, EncodedWeight[][]>> message) {
        
    	
    	HashMap<String, EncodedWeight[][]> encWeights = null;
    	HashMap<String, EncodedWeight[][]> encBias = null;
    	
    	if(message.size()>0)
        {
    		encWeights = (HashMap<String, EncodedWeight[][]>)message.get(0);
    		
    		boolean shall_we_remove = aggregator.removeWeight();
    		
        	aggregator.setEncodedWeights(encWeights, shall_we_remove);
            if(message.size()>1)
            {
            	encBias = (HashMap<String, EncodedWeight[][]>)message.get(1);
            	aggregator.setEncodedBias(encBias, shall_we_remove);
            }
            
        }
    	
    
    	
    	aggregator.decreaseCounter();
    	
    }

   
}
