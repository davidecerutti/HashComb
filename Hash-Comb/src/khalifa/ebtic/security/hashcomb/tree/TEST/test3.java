package khalifa.ebtic.security.hashcomb.tree.TEST;

import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import khalifa.ebtic.security.hashcomb.tree.CsvParserSimple;
import khalifa.ebtic.security.hashcomb.tree.Node;
import khalifa.ebtic.security.hashcomb.tree.Tree;
import khalifa.ebtic.security.hashcomb.tree.Utils;

public class test3 {
	
	
	public static String dir = "etc";
	public static String  fileN = "test1.csv";
	public static int channels=16;
	public static double min=-0.97484635830;
	public static double max=0.93763543220;
	//public static double min=-15;
	//public static double max=+15;
			
	
	public static void main(String[] args) throws Exception
	{
		double[] ds = {0.49918359643524224, 0.49167995128768793, 0.491703208314411955, 0.49901616284510925, 0.49901303289550886, 0.49451723101353867, 0.24442421374428364};
		
		boolean isHashed = true;
		
		
		
		long begin = System.currentTimeMillis();
		Tree tree = new Tree(channels, max, min);
		long end = System.currentTimeMillis();      
        //Tree tree = new Tree(channels, max, min, roud);
		long time = end-begin;
	    System.out.println();
	    System.out.println("Elapsed Time: "+time +" milli seconds");
	    
	    
		System.out.println("Binary Tree Example");
		System.out.println("Building tree with root value ");
		System.out.println("min: "+min+"    max: "+max);
		System.out.println("channels: "+channels);
		System.out.println("Traversing tree in order");
		int hashesNum = tree.traverseLevelOrder(isHashed);
		System.out.println("\n");
		System.out.println("Total Number of hashes: "+hashesNum);
		System.out.println("\n\n");
		
		
		
		for(int i=0; i<ds.length; i++)
			System.out.println(ds[i]+" --> "+tree.getHValues(ds[i], isHashed));
			
		
		Utils.writeHashTable2File("test2.ser", tree);
		
		HashMap<String, Node> myMap = Utils.readHashTable2File("test2.ser");
		
		
		if(isHashed)
		{
			double counter1 = 0;
			double counter2 = 0;
			for(int i=0; i<ds.length; i++)
			{
				counter1 = counter1 + ds[i];
				begin = System.currentTimeMillis();
				Vector<String> hashStrings = tree.getHValues(ds[i], isHashed);
				end = System.currentTimeMillis();   
				System.out.println("Element "+i+" searched in : "+time +" milli seconds");
				String key = hashStrings.elementAt(channels-1);
				Node node = myMap.get(key);
				double center = node.getCenter();
				counter2 = counter2 + center;
				System.out.println("Key: "+key+"     center: "+center+"    "+node.toString());
			}
			
			double average = counter1/ds.length;
			double aprox = counter2/ds.length;
			System.out.println("The real average is        "+average);
			System.out.println("The aproximated average is "+aprox);
			System.out.println("The LOSS is "+(average - aprox));
		}
		
		
	}

}
