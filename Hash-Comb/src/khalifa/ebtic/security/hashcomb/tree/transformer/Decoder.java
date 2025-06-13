package khalifa.ebtic.security.hashcomb.tree.transformer;

import java.util.HashMap;
import java.util.Vector;

import khalifa.ebtic.security.hashcomb.tree.Node;
import khalifa.ebtic.security.hashcomb.tree.Tree;
import khalifa.ebtic.security.hashcomb.tree.Utils;

public class Decoder {
	


	private HashMap<String, Node> myMap;
	
	public Decoder()
	{
		this.myMap = Utils.readHashTable2File("configuration.ser");
	}
	
	
	public double decode(String encValue)
	{
		
		Node node = myMap.get(encValue);
		double center = node.getCenter();
		return center;
	}
	
	
}