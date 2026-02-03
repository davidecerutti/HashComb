package khalifa.ebtic.security.hashcomb.tree;

import java.io.File;
import java.io.PrintWriter;
import java.security.PublicKey;
import java.security.spec.ECGenParameterSpec;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import javax.sound.midi.VoiceStatus;

public class runner 
{
	
	public static int channels=5;
	
	
	public static String dir = "etc";
	//public static String fileN = "spam_hf_6_original.P.csv";
	
	public static String  fileN = "spam_hf_rl_10_6.P.csv";
	//public static String  fileN = "spam_hf_wf_6.P.csv";
	//public static String  fileN = "test1.csv";
	
	public static double[] get_min_max_lenght(List<String[]> result)
	{
		double min=0, max=0;
		int lenght=0;
		
		Iterator<String[]> iter = result.iterator();
		
		while(iter.hasNext())
		{
			String[] aux = iter.next();
			for(int i=0; i<aux.length; i++)
			{
				//System.out.println(aux[i]);
				if(aux.length > lenght)
					lenght = aux.length;
				double value = Double.valueOf(aux[i]);
				
				if(value<min)
					min=value;
				if(value>max)
					max=value;
				
			}
		}
		
		return new double[]{min, max, lenght};
		
	}
	
	

	
	public static void main(String[] args) throws Exception
	{
		double min = -1.0;
		double max = 1.0;
		int lenght = 0;
		boolean isHashed=true;
		boolean lastOnly=true;
		
		int round=2;
		
        File file = new File(dir+"\\input\\"+fileN);
        
        
        CsvParserSimple obj = new CsvParserSimple();
        List<String[]> result = obj.readFile(file, 1);
        
        
        //this cose select (for each training the min and max values)
        double[] edges = get_min_max_lenght(result);
        min=edges[0];
        max=edges[1];
        lenght=(int) edges[2];
        
        
        min=min-0.00000001;
        max=max+0.00000001;
        
      //when enabled all the training are set with default (min,max)==(-1, +1)
      //  min = -1.50;
      //  max = 1.50;
        
     
        
        
        
        System.out.println("Min: "+min+"    Max: "+max+"   Lenght: "+lenght);
        
        Tree tree = new Tree(channels, max, min);
        //Tree tree = new Tree(channels, max, min, round);

		System.out.println("Binary Tree Example");
		System.out.println("Building tree with root value ");

		System.out.println("Traversing tree in order");
		tree.traverseLevelOrder(isHashed);
		System.out.println("\n\n");
		
		
		
		
		String outFileString = dir+"\\output\\"+"HC_"+channels+fileN;
		
		Utils.writeHashTable2File(outFileString+".ser", tree);
		
		PrintWriter writer = Utils.openCSV(outFileString, lenght, channels);
        
		Iterator<String[]> rows = result.iterator();
		while(rows.hasNext())
		{
			String[] rowStrings = rows.next();

			Utils.print2File2(writer, rowStrings, lenght, tree, isHashed, lastOnly);
		}
		
		writer.close();
	}
	

}
