package ebtic.labs.Federated.hashComb;

import java.util.HashMap;
import java.util.Iterator;

import khalifa.ebtic.security.hashcomb.tree.Node;
import khalifa.ebtic.security.hashcomb.tree.Tree;
import khalifa.ebtic.security.hashcomb.tree.Utils;

public class HCDecoder {
	
	HashMap<String, Node> myMap;
	
	public HCDecoder(String filename) 
	{
		this.myMap = Utils.readHashTable2File(filename);
		
	}

	public HCDecoder(Tree tree) 
	{
		HashMap<String, Node> hash = new HashMap<String, Node>(); 

		Node node = tree.getroot();


		hash = Utils.createHashEntry(hash, node.getLeft());
		hash = Utils.createHashEntry(hash, node.getRight());
		
		this.myMap = hash;
	}

	
	public double decode(String hash) 
	{

//		System.out.println(myMap.size()+"  --> "+hash);
		Node node = myMap.get(hash);
		double center = node.getCenter();
//		System.out.println(center+"\n");
		return center;
	}
	
}
