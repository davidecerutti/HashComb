package ebtic.labs.Federated.communications;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import ebtic.labs.Federated.threads.FinalModel;
import ebtic.labs.Federated.utils.EncodedWeight;
import ebtic.labs.utils.Matrix;

public class Client {
    
	private String host;
	private int port;
	boolean encoded;
	
	
	public Client(String host, int port)
	{
		this.host = host;
		this.port = port;
		
	}
	
	
	
	public ArrayList<HashMap<String, double[][]>> sendWeights(ArrayList<HashMap<String, double[][]>> allWeights) throws UnknownHostException,
            IOException, ClassNotFoundException {
      
		this.encoded = false;
		
		HashMap<String, double[][]> weights = null;
		HashMap<String, double[][]> bias = null;
		
		if(allWeights.size()>=1)
		{
			weights = allWeights.get(0);
			if(weights.size()>1)
				bias = allWeights.get(1);
		}
		
		ArrayList<HashMap<String, double[][]>> finalW = (ArrayList<HashMap<String, double[][]>>) this.sendMessage(weights, bias);
		return finalW;
    }
	


	public ArrayList<HashMap<String, EncodedWeight[][]>> sendEncodedWeights(ArrayList<HashMap<String, EncodedWeight[][]>> allWeights) throws UnknownHostException, ClassNotFoundException, IOException  {
		
		this.encoded = true;
		HashMap<String, EncodedWeight[][]> encodedWeights = null;
		HashMap<String, EncodedWeight[][]> encodedBias = null;
		
		if(allWeights.size()>=1)
		{
			encodedWeights = allWeights.get(0);
			if(allWeights.size()>1)
				encodedBias = allWeights.get(1);
		}
		
		ArrayList<HashMap<String, EncodedWeight[][]>> finalW =(ArrayList<HashMap<String, EncodedWeight[][]>>) this.sendEncodedMessage(encodedWeights, encodedBias);
		
		return finalW;
	}
	
	

	private ArrayList<HashMap<String, double[][]>> sendMessage(HashMap<String, double[][]> weights, HashMap<String, double[][]> bias) throws UnknownHostException,
	IOException, ClassNotFoundException 
	{

		ArrayList<HashMap> woi=new ArrayList<>();
		woi.add(weights);
		if(bias!=null)
			woi.add(bias);
		System.out.println("Client connecting to "+this.host+":"+this.port);
		Socket socket = new Socket(this.host, this.port);
		System.out.println("Client connected");
		ObjectOutputStream os = new ObjectOutputStream(socket.getOutputStream());
		System.out.println("Ok");
		os.writeObject(woi);
		System.out.println("Envoi des informations au serveur ...");
		ObjectInputStream is = new ObjectInputStream(socket.getInputStream());
		ArrayList<HashMap<String, double[][]>> modelWeights  = (ArrayList<HashMap<String, double[][]>>) is.readObject();	
		System.out.println("return Message is = " + modelWeights);
		socket.close();
		return modelWeights;				
		
	}
	

	
	private ArrayList<HashMap<String, EncodedWeight[][]>> sendEncodedMessage(HashMap<String, EncodedWeight[][]> encodedWeights, HashMap<String, EncodedWeight[][]> encodedBias) throws UnknownHostException,
	IOException, ClassNotFoundException 
	{

		ArrayList<HashMap<String, EncodedWeight[][]>> woi=new ArrayList<>();
		woi.add(encodedWeights);
		if(encodedBias!=null)
			woi.add(encodedBias);

		System.out.println("Client connecting to "+this.host+":"+this.port);
		Socket socket = new Socket(this.host, this.port);
		System.out.println("Client connected");
		ObjectOutputStream os = new ObjectOutputStream(socket.getOutputStream());
		System.out.println("Ok");
		os.writeObject(woi);
		System.out.println("Envoi des informations au serveur ...");
		ObjectInputStream is = new ObjectInputStream(socket.getInputStream());
		ArrayList<HashMap<String, EncodedWeight[][]>> modelWeights  = (ArrayList<HashMap<String, EncodedWeight[][]>>) is.readObject();
		System.out.println("return Message is = " + modelWeights);
		socket.close();
		return modelWeights;				

	}
}