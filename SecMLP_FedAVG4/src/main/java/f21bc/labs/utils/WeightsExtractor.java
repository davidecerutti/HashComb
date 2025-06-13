package f21bc.labs.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import f21bc.labs.NN.Layer;

public class WeightsExtractor {


	private PrintWriter writer = null;
	public ArrayList<Layer> Layers;

	private boolean enable = false;
	private int index;
	private String fileName;
	private int stop = -1;
	private int counter = 0;
	
	
	public WeightsExtractor(ArrayList<Layer>  layers)
	{
		this.Layers = layers;
	}
	
	public WeightsExtractor(int epochs)
	{
		this.stop = epochs;
	}

	
	public void setLayers(ArrayList<Layer>  layers)
	{
		this.Layers = layers;
	}
	
	
	public void setFileName(String fileName, int index, boolean enable)
	{
		this.fileName = fileName;
		this.index = index;
		this.enable = enable;
	}
	
	
	public void setStop(int stop)
	{
		this.stop = stop;
	}
	
	public PrintWriter openCSV()
			throws IOException
	{

		if(index< this.Layers.size()&&enable)
		{
			Layer L = this.Layers.get(index);
			double[][] m = L.get_W();
			String header = WeightsExtractor.header(m);
			System.out.println("Trying to open file: "+this.fileName);
			File file = new File(this.fileName);
			if(!file.exists())
				try
			{
					file.createNewFile();
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println("Please make sure all the subdirectories exists");
				System.exit(0);
			}

			FileWriter fileWriter = new FileWriter(fileName, true);
			PrintWriter printWriter = new PrintWriter(fileWriter);
			System.out.println("Printing Header: "+header);
			
			//printWriter.append(header+System.lineSeparator());
			printWriter.append(header);
			this.writer = printWriter;
			return printWriter;
		}
		return this.writer;

	}




	public static String header(double[][] m)
	{
		String out = "";
		for(int i=0; i<m.length; i++)
		{	String separator=",";
			if(i==0)
				separator="";
			double[] aux = m[i];
			for(int j=0; j<aux.length; j++)
			{
				out = out +separator+"w_"+i+"_"+j;
				separator=",";
			}
				
		}

		out = out + "\n";
		return out;
	}

	
	/**
	 * The method implement the representaton of the content of the matrix
	 * @param m input matrix
	 * @return the string representing the input
	 */
	public static String toString(double[][] m)
	{
		
		String out = "";
		
		for(int i=0; i<m.length; i++)
		{
			String separator=",";
			if(i==0)
				separator="";
			double[] aux = m[i];
			for(int j=0; j<aux.length; j++)
			{
				out = out +separator+aux[j];
				separator=",";
			}
				
			
		}
		out = out + "\n";
		return out;
	}

	
	
	public void print2File()
	{
		if((this.enable)&&(this.counter <= this.stop))
		{
			String out = "";
			Layer L = this.Layers.get(this.index);
			double[][] m = L.get_W();
			
			out = WeightsExtractor.toString(m);
			//System.out.println(out);
			//this.writer.append(out+System.lineSeparator());
			this.writer.append(out);
		}
		
//		else {
//			System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");	
//		}
		
		this.counter= this.counter + 1;
	}

	public static void main(String[] args)
	{

	}

}
