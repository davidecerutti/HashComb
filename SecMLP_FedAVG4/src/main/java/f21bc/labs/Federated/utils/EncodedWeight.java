package f21bc.labs.Federated.utils;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;

public class EncodedWeight implements Serializable{
	
	private static final long serialVersionUID = -2061918465734780571L;
	
	private HashMap<String, Double> weightMap = new HashMap();
	

	public void setWeights(HashMap<String, Double> newWeights)
	{
		this.weightMap = newWeights;
	}
	
	
	public void addWeight(String weight)
	{
		if(weightMap.get(weight) != null)
		{
			double counter = weightMap.get(weight) + 1;
			weightMap.put(weight, counter);
		}
		
		else {
			
			weightMap.put(weight, (double) 1);
		}
	}
	
	public void removeWeight(String weight)
	{
		weightMap.remove(weight);
	}
	
	
	public void replaceWeight(String weight, double newValue)
	{
			weightMap.put(weight, newValue);
		
	}
	
	
	public void addWeight(EncodedWeight encWeight)
	{

		Iterator<String> iter = encWeight.weightMap.keySet().iterator();
		while(iter.hasNext())
		{
			
			String code = iter.next();
			
			this.addWeight(code);
		}

	}
	
	public double getItems(String hash)
	{
		return this.weightMap.get(hash);
	}
	
	public Iterator<String> getKeys()
	{
		if(!this.weightMap.isEmpty())
			return this.weightMap.keySet().iterator();
		else 
		{
			Iterator<String> Itr = Collections.emptyIterator();
			return Itr;
		} 
	}
	
	public String toString()
	{
		String out = "";
		Iterator<String> iter = this.getKeys();
		while(iter.hasNext())
		{
			String hash = iter.next();
			double counter = this.getItems(hash);
			out=out+counter+"*("+hash+")  ";
		}
		return out;
	}

}
