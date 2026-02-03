package ebtic.labs.Federated;

import java.util.HashMap;
import java.util.Iterator;

import org.w3c.dom.css.Counter;

import ebtic.labs.Federated.utils.EncodedWeight;
import ebtic.labs.utils.CoinToss;
import ebtic.labs.utils.Matrix;

public class ModelAggregator {
	
	
	private boolean encoded=false;
	private int nodes;
	private int randomNodes;
	private boolean randomSetting = false;
	private CoinToss toss = new CoinToss();
	private boolean firstW = true;
	private boolean firstb = true;
	
	
	private HashMap<String, double[][]> weights = new HashMap<String, double[][]>();
	private HashMap<String, double[][]> finalWeights = new HashMap<String, double[][]>();;
	private HashMap<String, EncodedWeight[][]> encodedWeights = new HashMap<String, EncodedWeight[][]>();
	
	
	private HashMap<String, double[][]> bias = new HashMap<String, double[][]>();
	private HashMap<String, double[][]> finalBias = new HashMap<String, double[][]>();;
	private HashMap<String, EncodedWeight[][]> encodedBias = new HashMap<String, EncodedWeight[][]>();
	
	public int counter;
	public boolean isCompleted=false;
	private int iteration =0;
	
	public ModelAggregator(boolean encoded)
	{
		this.encoded = encoded;
		
		
	}

	
	public void setRandomness()
	{
		System.out.println("Setting random clients...");
		this.randomSetting = true;
	}
	
	public HashMap<String, double[][]> getFinalWeights() 
	{
		return this.finalWeights;
	} 
	
	public void setFinalWeights(HashMap<String, double[][]> finalweights) 
	{
		this.weights = finalweights;
		this.finalWeights = finalweights;
	}
	
	
	public void setFinalEncodedWeights(HashMap<String, EncodedWeight[][]> encodedWeights2) {
		// TODO Auto-generated method stub
		this.encodedWeights = encodedWeights2;
	}
	
	public HashMap getBias()
	{
		if(this.encoded)
			return this.encodedBias;
		else return this.bias;
	}
	
	public HashMap<String, double[][]> getFinalBias() 
	{
		return this.finalBias;
	} 
	
	public void setFinalBias(HashMap<String, double[][]> finalweights) 
	{
		this.bias = finalweights;
		this.finalBias = finalweights;
	}
	
	
	public void setFinalEncodedBias(HashMap<String, EncodedWeight[][]> encodedWeights2) {
		// TODO Auto-generated method stub
		this.encodedBias = encodedWeights2;
	}
	
	
	public int getNodes()
	{
		return this.nodes;
	}
	
	public int getRandomNodes()
	{
		return this.randomNodes;
	}
	
	public void setCounter(int nodes) 
	{
		System.out.println("reset the counter...");
		this.counter = nodes;
		this.nodes = nodes;
		this.randomNodes = nodes;
		this.isCompleted=false;
		this.iteration = 0;
		this.firstW = true;
		this.firstb = true;
	}
	
	public void setNotFirstW()
	{
		this.firstW = false;
	}
	
	public void setNotFirstb()
	{
		this.firstb = false;
	}
	
	public void decreaseCounter() 
	{
		this.counter --;
		this.iteration++;
	}
	
	
	public boolean isFinilazed()
	{
		return (this.counter==0);
	}
	
	public HashMap getWeights()
	{
		if(this.encoded)
			return this.encodedWeights;
		else return this.weights;
	}
	
	
	public boolean removeWeight()
	{
		if(toss.flip()==CoinToss.Coin.Heads)
			return true;
		else return false;
	}
	
	
	public void setWeights(HashMap<String, double[][]> individualWeights, boolean dropout)
	{
		boolean shall_we_remove = false;
		if(this.randomSetting)
			shall_we_remove = dropout;
		System.out.println("We are removing: "+shall_we_remove+"   aggregator hash "+this.hashCode());
		if((!individualWeights.isEmpty())&&(!shall_we_remove))
		{
			Iterator<String> iter = individualWeights.keySet().iterator();
			while(iter.hasNext())
			{
				String layer = iter.next();
				
				double[][] W = individualWeights.get(layer);
				
				if((weights.get(layer) != null) && (this.iteration>0) && !this.firstW)
				{
					double[][] aux = weights.get(layer);
					int[] shapeA = Matrix.getShape(aux);
					for(int i =0; i< shapeA[0]; i++)
					{
						for(int j=0; j<shapeA[1]; j++)
						{
							double d = aux[i][j];
							
							aux[i][j] = aux[i][j] + W[i][j];
							
							if((i==1)&&(j==0))
								System.out.println("Thread "+counter+" Layer: "+layer+"  "+i+"x"+j+" old: "+d+" add: "+W[i][j]+"  new: "+aux[i][j]);		
						}
							
				
					}
					
					weights.put(layer, aux);
					
				}
				
				else 
				{
					weights.put(layer, W);
					System.out.println("Thread "+counter+" Layer: "+layer+"  "+1+"x"+0+" old: "+W[1][0]+"  new: "+W[1][0]);
				}
					
				}
			this.setNotFirstW();
		}
		else
		{
			this.randomNodes = this.randomNodes -1;
			System.out.println("Removing a client... total clients:"+this.randomNodes);
		}
			
	}
	
	
	public void setBias(HashMap<String, double[][]> individualBias, boolean dropout)
	{
		boolean shall_we_remove = false;
		if(this.randomSetting)
			shall_we_remove = dropout;
		
		if((!individualBias.isEmpty())&&(!shall_we_remove))
		{
			Iterator<String> iter = individualBias.keySet().iterator();
			while(iter.hasNext())
			{
				String layer = iter.next();
				
				double[][] b = individualBias.get(layer);
				
				if((bias.get(layer) != null) && (this.iteration>0) && !this.firstb)
				{
					double[][] aux = bias.get(layer);
					int[] shapeA = Matrix.getShape(aux);
					for(int i =0; i< shapeA[0]; i++)
					{
						for(int j=0; j<shapeA[1]; j++)
						{
							double d = aux[i][j];
							aux[i][j] = aux[i][j] + b[i][j];
							if((i==1)&&(j==0))
								System.out.println("Thread "+counter+" (b)  Layer: "+layer+"  "+i+"x"+j+" old: "+d+" add: "+b[i][j]+"  new: "+aux[i][j]);		
						}
							
				
					}
					
					bias.put(layer, aux);
					
				}
				
				else 
				{
					bias.put(layer, b);
					
					System.out.println("Thread "+counter+" (b)  Layer: "+layer+"  "+1+"x"+0+" old: "+b[0][0]+"  new: "+b[0][0]);
				}
					
				}
			this.setNotFirstb();
			}
	}
	
	
	
	public void setEncodedWeights(HashMap<String, EncodedWeight[][]> individualWeights, boolean dropout)
	{
		boolean shall_we_remove = false;
		if(this.randomSetting)
			shall_we_remove = dropout;
		System.out.println("We are removing: "+shall_we_remove+"   aggregator hash "+this.hashCode());
		if((!individualWeights.isEmpty())&&(!shall_we_remove))
		{
			Iterator<String> iter = individualWeights.keySet().iterator();
			while(iter.hasNext())
			{
				String layer = iter.next();
				
				EncodedWeight[][] W = individualWeights.get(layer);
				
				if((encodedWeights.get(layer) != null) && (this.iteration>0) && !this.firstW)
				{
					EncodedWeight[][] aux = encodedWeights.get(layer);
					int[] shapeA = Matrix.getShape(aux);
					for(int i =0; i< shapeA[0]; i++)
					{
						for(int j=0; j<shapeA[1]; j++)
						{
							aux[i][j].addWeight(W[i][j]);
							
							//System.out.println("Thread "+counter+" Layer: "+layer+"  "+i+"x"+j+" add: "+W[i][j]+"  new: "+aux[i][j]);
						}
							
						
				
					}
					
					encodedWeights.put(layer, aux);
					
				}
				
				else 
				{
					encodedWeights.put(layer, W);
					
					//System.out.println("Thread "+counter+" Layer: "+layer+"  "+1+"x"+0+" old: "+W[1][0]+"  new: "+W[1][0]);
				}
					
			}
			this.setNotFirstW();
		}
		else
		{
			this.randomNodes = this.randomNodes -1;
			System.out.println("Removing a client... total clients:"+this.randomNodes);
		}
			
	}


	
	public static String printWeightSize(HashMap<String, double[][]> weights, boolean bias)
	{
		String out = "";
		String type = "W";
		if(bias)
			type="b";
		Iterator<String> iter = weights.keySet().iterator();
		while(iter.hasNext())
		{
			String k = iter.next();
			double[][] w = weights.get(k);
			int[] shape = Matrix.getShape(w);
			out = out + type+"_Layer "+k+" shape: "+shape[0]+"x"+shape[1]+".\n";
		}
		
		return out;
		
		
	}

	
	public void setEncodedBias(HashMap<String, EncodedWeight[][]> individualBias, boolean dropout)
	{
		boolean shall_we_remove = false;
		if(this.randomSetting)
			shall_we_remove = dropout;
		if((!individualBias.isEmpty())&&(!shall_we_remove))
		{
			Iterator<String> iter = individualBias.keySet().iterator();
			while(iter.hasNext())
			{
				String layer = iter.next();
				
				EncodedWeight[][] b = individualBias.get(layer);
				
				if((this.encodedBias.get(layer) != null) && (this.iteration>0) && !this.firstb)
				{
					EncodedWeight[][] aux = this.encodedBias.get(layer);
					int[] shapeA = Matrix.getShape(aux);
					for(int i =0; i< shapeA[0]; i++)
					{
						for(int j=0; j<shapeA[1]; j++)
						{
							aux[i][j].addWeight(b[i][j]);
							//System.out.println("Thread "+counter+" (b)  Layer: "+layer+"  "+i+"x"+j+" add: "+b[i][j]+"  new: "+aux[i][j]);
						}
							
						
				
					}
					
					this.encodedBias.put(layer, aux);
					
				}
				
				else 
				{
					this.encodedBias.put(layer, b);
					
					//System.out.println("Thread "+counter+" (b)  Layer: "+layer+"  "+1+"x"+0+" old: "+b[0][0]+"  new: "+b[0][0]);
				}
					
			}
			this.setNotFirstb();
		}
	}
	
}
