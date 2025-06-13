package f21bc.labs.objects;

import java.io.FileNotFoundException;

import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Stack;

import org.apache.commons.lang3.math.NumberUtils;

import com.opencsv.CSVReader;
import com.opencsv.bean.ColumnPositionMappingStrategy;
import com.opencsv.bean.CsvBindByPosition;
import com.opencsv.bean.CsvToBean;
import com.opencsv.bean.CsvToBeanBuilder;
import com.opencsv.exceptions.CsvValidationException;

public class Generic extends MyObject{
		

	
	private double[] data;
	private int outcome;

	@Override
	public double[] returnData() {
		// TODO Auto-generated method stub
		
		return this.data;
		
	}


	@Override
	public int returnLabel() {
		// TODO Auto-generated method stub
		return this.outcome;
	}


	
	public Generic(double[] data, int outcome)
	{
		this.data=data;
		this.outcome=outcome;
	}
	
	
	@Override
	public String toString() 
	{
		String output = "{ ";
		for(int i=0; i<this.getDataSize(); i++)
		{
			output+="Property"+i+": "+this.data[i];
			output+=", ";
		}	
		output+="Outcome: "+this.outcome;
		output+=" }";
		
		return output;
	}
	
	
	public static List<MyObject> instanciateCSV(String fileName) throws IllegalStateException, FileNotFoundException
	{
		
		String[] record;
		List<MyObject> beans = new Stack<MyObject>();
		CSVReader csvReader = new CSVReader (new FileReader(fileName));
		try {
			while ((record = csvReader.readNext()) != null) {
			    
				
				double[] data = new double[record.length-1];
				for(int i=0; i<(record.length-1); i++)
				{
					data[i] = NumberUtils.toDouble(record[i]);
				}
				int outcome = NumberUtils.toInt(record[record.length-1]);
				Generic obj = new Generic(data, outcome);
				beans.add(obj);
//				System.out.println(obj.toString());
				
			}
		} catch (CsvValidationException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        
  
        
        return beans;
        
	}


	public int getDataSize() {
		// TODO Auto-generated method stub
		return this.data.length;
	}
	
}
