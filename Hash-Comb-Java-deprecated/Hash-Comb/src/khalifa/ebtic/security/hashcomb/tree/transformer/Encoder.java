package khalifa.ebtic.security.hashcomb.tree.transformer;

import java.util.HashMap;
import java.util.Vector;

import khalifa.ebtic.security.hashcomb.tree.Node;
import khalifa.ebtic.security.hashcomb.tree.Tree;
import khalifa.ebtic.security.hashcomb.tree.Utils;

public class Encoder {
	
	private static int channels;
	private static double min;
	private static double max;
	
	private Tree tree;

	private HashMap<String, Node> myMap;
	
	public Encoder(int channels, double min, double max)
	{
		this.channels = channels;
		this.min = min;
		this.max= max;
		this.tree = new Tree(channels, max, min);
		Utils.writeHashTable2File("configuration.ser", this.tree);
		this.myMap = Utils.readHashTable2File("configuration.ser");
	}
	
	
	
	
	public String encode(double value)
	{
		
		Vector<String> hashStrings = tree.getHValues(value, true);
		String key = hashStrings.elementAt(channels-1);
		return key;
	}
	
	
}
