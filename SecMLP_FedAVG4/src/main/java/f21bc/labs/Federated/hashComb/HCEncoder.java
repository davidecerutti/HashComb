package f21bc.labs.Federated.hashComb;

import java.util.Vector;

import khalifa.ebtic.security.hashcomb.tree.Tree;
import khalifa.ebtic.security.hashcomb.tree.Utils;

public class HCEncoder {
	
	
	private Tree tree;
	private boolean isHash;
	
	public HCEncoder(int channels, double min, double max, boolean hashed)
	{
		this.tree = new Tree(channels, max, min);
		this.isHash = hashed;
	}
	
	
	public Vector<String> encode(double value) 
	{
		if(value<this.tree.getMin())
			value = this.tree.getMin();
		if(value>this.tree.getMax())
			value = this.tree.getMax();
		
		return this.tree.getHValues(value, this.isHash);
	}
	
	public String getLastChannelValue(Vector<String> hash)
	{
		return hash.elementAt(this.tree.getChannels()-1);
	}
	
	public void store(String filename)
	{
		Utils.writeHashTable2File(filename, this.tree);
	}
	
	public Tree getTree()
	{
		return this.tree;
	}
}
