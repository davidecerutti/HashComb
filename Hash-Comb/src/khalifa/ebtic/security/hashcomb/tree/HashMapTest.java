package khalifa.ebtic.security.hashcomb.tree;

import java.util.HashMap;
import java.util.Vector;

public class HashMapTest {
	
	
	public static void main(String[] args)
	{
		
		double min=-0.49921549185919006;
		double max=0.6309332106577579;
		int channels=8;
		
		Tree tree = new Tree(channels, max, min);
		
		double number = 0.3249847613254643;
		
		System.out.println("my number: "+number);
		
		Vector<String> hashes = tree.getHValues(number, true);
		System.out.println(hashes);
		
		String key = hashes.get(hashes.size()-1);
		
		System.out.println("key: "+key);		
				
		Utils.writeHashTable2File("hasmap.ser", tree);
		
		HashMap<String, Node> myMap = Utils.readHashTable2File("hasmap.ser");
		
		 
		
		Node node = myMap.get(key);
		
		System.out.println(node.min+"    "+node.max);
		System.out.println("Approximation: "+Utils.GetAvarage(node.min, node.max));
		
	}

}
