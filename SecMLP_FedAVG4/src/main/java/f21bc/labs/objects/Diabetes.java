package f21bc.labs.objects;

import java.io.FileNotFoundException;

import java.io.FileReader;
import java.util.List;

import org.apache.commons.lang3.math.NumberUtils;

import com.opencsv.bean.CsvBindByPosition;
import com.opencsv.bean.CsvToBeanBuilder;

public class Diabetes extends MyObject{
		
	@CsvBindByPosition(position = 0)
    private String Pregnancies;

    @CsvBindByPosition(position = 1)
    private String Glucose;

    @CsvBindByPosition(position = 2)
    private String BloodPressure;

    @CsvBindByPosition(position = 3)
    private String SkinThickness;
	

    @CsvBindByPosition(position = 4)
    private String Insulin;

    @CsvBindByPosition(position = 5)
    private String BMI;

    @CsvBindByPosition(position = 6)
    private String DiabetesPedigreeFunction;

    @CsvBindByPosition(position = 7)
    private String Age;
    
    
    @CsvBindByPosition(position = 8)
    private String Outcome;


	@Override
	public double[] returnData() {
		// TODO Auto-generated method stub
		
		double[] output = new double[8];
		
		
		output[0] = NumberUtils.toInt(this.Pregnancies);
		output[1] = NumberUtils.toInt(this.Glucose);
		output[2] = NumberUtils.toInt(this.BloodPressure);
		output[3] = NumberUtils.toInt(this.SkinThickness);
		output[4] = NumberUtils.toInt(this.Insulin);
		output[5] = NumberUtils.toDouble(this.BMI);
		output[6] = NumberUtils.toDouble(this.DiabetesPedigreeFunction);
		output[7] = NumberUtils.toInt(this.Age);
		return output;
		
	}


	@Override
	public int returnLabel() {
		// TODO Auto-generated method stub
//		return Integer.parseInt(Outcome);
		String out = Outcome;
		int num = NumberUtils.toInt(out);
		return num;
	}


	
	
	
	@Override
	public String toString() 
	{
		// TODO Auto-generated method stub
		String output = "{ ";
		output+="Pregnancies: "+this.Pregnancies;
		output+=", ";
		output+="Glucose: "+this.Glucose;
		output+=", ";
		output+="BloodPressure: "+this.BloodPressure;
		output+=", ";
		output+="SkinThickness: "+this.SkinThickness;
		output+=", ";
		output+="Insulin: "+this.Insulin;
		output+=", ";
		output+="BMI: "+this.BMI;
		output+=", ";
		output+="DiabetesPedigreeFunction: "+this.DiabetesPedigreeFunction;
		output+=", ";
		output+="Age: "+this.Age;
		output+=", ";
		output+="Outcome: "+this.Outcome;
		output+=" }";
		
		return output;
			
	}
	
	public static List<MyObject> instanciateCSV(String fileName) throws IllegalStateException, FileNotFoundException
	{
		List<MyObject> beans = new CsvToBeanBuilder(new FileReader(fileName))
                .withType(Diabetes.class)
                .build()
                .parse();

        beans.remove(0);
        beans.forEach(System.out::println);
        
        return beans;
	}


	public int getDataSize() {
		// TODO Auto-generated method stub
		return 8;
	}
	
}
