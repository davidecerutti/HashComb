package khalifa.ebtic.security.hashcomb.tree.TEST;

import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import khalifa.ebtic.security.hashcomb.tree.CsvParserSimple;
import khalifa.ebtic.security.hashcomb.tree.Node;
import khalifa.ebtic.security.hashcomb.tree.Utils;

public class test1 {
	
	
	public static String dir = "etc";
	public static String  fileN = "test1.csv";
	public static int channels=3;
	
	public static void main(String[] args) throws Exception
	{
	
		
		
		String inputFile = dir+"\\output\\"+"HC_"+channels+fileN;
		
		File file = new File(inputFile);
		
		CsvParserSimple obj = new CsvParserSimple();
        List<String[]> result = obj.readFile(file, 1);
        
        
        
        HashMap<String, Node> myMap = Utils.readHashTable2File(inputFile+".ser");
        
        PrintWriter writer = Utils.openCSV(inputFile+".dec", 19, channels);
        
        Iterator<String[]> rows = result.iterator();
		while(rows.hasNext())
		{
			String[] rowStrings = rows.next();
			
			Utils.decode2File(writer, rowStrings, myMap);
		}
		
		writer.close();
	}

}
