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

public class test2 {
	
	
	public static String dir = "etc";
	public static String  fileN = "test1.csv";
	public static int channels=4;
	public static double min=-1;
	public static double max=1;
	
//	public static double min=-0.97484635830;
//	public static double max=0.93763543220;

	//public static double min=-15;
	//public static double max=+15;
			
	
	public static void main(String[] args) throws Exception
	{
		double[] ds = {0.49918359643524224, 0.47167995128768793, 0.491703208314411955, 0.49901616284510925, 0.48451723101353867};
		
		//int[] index = {channels-1, channels-1, channels-1,channels-1,channels-1};
		
		// L=16
		//int[] index = {10, 2, 8, 15, 5};
		int[] index = {3, 3, 4, 15, 15};
		
		boolean isHashed = false;
		
		
		
		
		Tree tree = new Tree(channels, max, min);
        //Tree tree = new Tree(channels, max, min, roud);

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
				Vector<String> hashStrings = tree.getHValues(ds[i], isHashed);
		//		System.out.println(hashStrings.elementAt(channels-1));
		//		String key = hashStrings.elementAt(channels-1);
				String key = hashStrings.elementAt(index[i]);
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
