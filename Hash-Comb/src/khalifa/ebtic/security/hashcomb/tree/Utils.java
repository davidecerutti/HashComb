package khalifa.ebtic.security.hashcomb.tree;

import java.awt.RenderingHints.Key;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import org.apache.commons.collections.functors.ForClosure;



public class Utils {
	
	
	
	private static String generateHeader(int features, int channels) 
	{
		
		String out = "";
		for(int i=0; i<features; i++)
		{	String separator=",";
			if(i==0)
				separator="";
			for(int j=0; j<channels; j++)
			{
				out = out +separator+"feat_"+i+"_"+j;
				separator=",";
			}
				
		}

		out = out + "\n";
		return out;
		
	}
	
	
	private static String generateHeader(int features) 
	{
		
		String out = "";
		for(int i=0; i<features; i++)
		{	String separator=",";
			if(i==0)
				separator="";
			
				out = out +separator+"feat_"+i;
				separator=",";
			
				
		}

		out = out + "\n";
		return out;
		
	}
	
	public static PrintWriter openCSV(String filename, int features, int channels)
			throws IOException
	{



		
		
		//String header = generateHeader(features, channels);
		String header = generateHeader(features);
		System.out.println("Trying to open file: "+filename);
		File file = new File(filename);
		if(!file.exists())
			try
		{
				file.createNewFile();
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Please make sure all the subdirectories exists");
			System.exit(0);
		}

		FileWriter fileWriter = new FileWriter(filename, true);
		PrintWriter printWriter = new PrintWriter(fileWriter);
		System.out.println("Printing Header: "+header);

		//printWriter.append(header+System.lineSeparator());
		printWriter.append(header);
		return printWriter;


	}


	public static void print2File(PrintWriter writer, String[] row, int lenght, Tree tree, boolean isHashed)
	{
		
			String out = "";
			int count = 0;
			int channels = 0;
			for(int s=0; s<lenght; s++)
			{
				String separator=",";
				if(s==0)
					separator="";
				String feature = "";
					if(row.length>s)
					{
						Vector<String> hashes = tree.getHValues(Double.parseDouble(row[s]), isHashed);
						if(channels==0)
							channels=hashes.size();
						for(int i=0; i<hashes.size(); i++)
						{
							
							String aux = hashes.get(i);
							//aux = s+"-"+aux;
							
							out = out +separator+aux;
							separator=",";
								
						}
						
					}
					
					else 
					{
						for(int i=0; i<channels; i++)
						{
							out = out +separator+0;
							separator=",";		
						}
					}
						
			}
			
			out = out + "\n";	
			//this.writer.append(out+System.lineSeparator());
			writer.append(out);
		
	}
	
	
	
	public static HashMap<String, Node> createHashEntry(HashMap<String, Node> hash, Node node)
	{
		
		String key = node.getValue(true);
		
		hash.put(key, node);
		
		if(!node.isLeaf())
		{
			Node leftNode = node.left;
			Node rightNode = node.right;
			
			hash = createHashEntry(hash, leftNode);
			hash = createHashEntry(hash, rightNode);
		}
		
		return hash;
		
	}
	
	
	
	public static double GetAvarage(double number1, double number2) {
		  double sum, avarage;
		  sum = number1 + number2;
		  avarage = sum / 2;
		  return avarage;
		}
	

	public static void writeHashTable2File(String fileName, Tree tree)
	{
		HashMap<String, Node> hash = new HashMap<String, Node>(); 

		Node node = tree.root;


		hash = createHashEntry(hash, node.left);
		hash = createHashEntry(hash, node.right);

		FileOutputStream fos;
		try {
			fos = new FileOutputStream(fileName);

			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(hash);
			oos.close();	

		} catch (FileNotFoundException e) {e.printStackTrace(); System.out.println(e.getMessage());
		} catch (IOException e) {e.printStackTrace(); System.out.println(e.getMessage());
		}

	}

	
	public static HashMap<String, Node> readHashTable2File(String fileName) 
	{
		try {
			FileInputStream fis = new FileInputStream(fileName);
			ObjectInputStream ois = new ObjectInputStream(fis);
			HashMap<String, Node> hash = (HashMap<String, Node>) ois.readObject();
			ois.close();
			return hash;
		} catch (FileNotFoundException e) {e.printStackTrace();
		} catch (IOException e) {e.printStackTrace();
		} catch (ClassNotFoundException e) {e.printStackTrace();
		}

		return null;
	}
	
	

	
	public static void print2File2(PrintWriter writer, String[] row, int lenght, Tree tree, boolean isHashed, boolean lastonly)
	{
		
		
			String out = "";
			int count = 0;
			int channels = 0;
			for(int s=0; s<lenght; s++)
			{
				String separator=",";
				if(s==0)
					separator="";
				String feature = "";
					if(row.length>s)
					{
						double weight = Double.parseDouble(row[s]);
						if(tree.isRounded)
							weight=Tree.round(weight, tree.places);
						Vector<String> hashes = tree.getHValues(weight, isHashed);
						if(channels==0)
							channels=hashes.size();
						String aux="";
						if(!lastonly)
						{
							for(int i=0; i<hashes.size(); i++)
							{
								
								String value = hashes.get(i);
								//aux = s+"-"+aux;
								if(i==0)
									aux = value;
								else
									aux = aux +":"+value;
										
							}
							
						}
						else
						{
							aux = getLastChannelValue(hashes);
						}
						
						
						out = out +separator+aux;
					}
					
					else 
					{
						out = out +separator+"null";
					}
					separator=",";
						
			}
			
			out = out + "\n";
			//System.out.println(out);
			//this.writer.append(out+System.lineSeparator());
			writer.append(out);
		
	}
	


	public static String getLastChannelValue(Vector<String> hash)
	{
		return hash.elementAt(hash.size()-1);
	}

	
	public static void decode2File(PrintWriter writer, String[] row, HashMap<String, Node> map)
	{
		
		
			String out = "";
			
			for(int s=0; s<row.length; s++)
			{
				String separator=",";
				if(s==0)
					separator="";
				String feature = "";
				String[] aux = row[s].split(":");;
				String key = aux[aux.length-1];	
				
				
				Node node = map.get(key);
				
				double average = Utils.GetAvarage(node.min, node.max);
				
				out = out +separator +average;
			}
			
			out = out + "\n";
			//System.out.println(out);
			//this.writer.append(out+System.lineSeparator());
			writer.append(out);
		
	}
}
