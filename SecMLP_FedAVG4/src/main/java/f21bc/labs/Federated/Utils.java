package f21bc.labs.Federated;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Properties;

import f21bc.labs.AF.ActivationFunction;
import f21bc.labs.Federated.Noise.GaussianNoise;
import f21bc.labs.Federated.Noise.NoiseGenerator;
import f21bc.labs.Federated.utils.EncodedWeight;

public class Utils {
	
	
	public static Properties loadConf(String filename)
	{
		Properties prop = new Properties();
        try (InputStream input = new FileInputStream(filename)) 
        {
            // load a properties file
            prop.load(input);

            // get the property value and print it out
            System.out.println(prop.getProperty("db.url"));
            System.out.println(prop.getProperty("db.user"));
            System.out.println(prop.getProperty("db.password"));

        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return prop;
	}
	
	
	public static String getModelClass(Properties prop)
	{
		return prop.getProperty("model.class");
	}
	
	
	public static String getDataFile(Properties prop)
	{
		return prop.getProperty("data.file");
	}
	
	public static String getInitWeightsFile(Properties prop)
	{
		return prop.getProperty("mlp.init.file");
	}
	
	
	public static String getEncodingFile(Properties prop)
	{
		return prop.getProperty("encoding.file");
	}
	
	public static String getGMServer_IP(Properties prop)
	{
		return prop.getProperty("global.server.ip");
	}
	

	public static int getGMServer_Port(Properties prop)
	{
		return Integer.parseInt(prop.getProperty("global.server.port"));
	}
	
	public static int getClasses(Properties prop)
	{
		try {return Integer.parseInt(prop.getProperty("model.classes"));}
		catch(Exception e)
		{
			e.printStackTrace();
			System.out.println("The number of classes for the classification is not specified, the default binary is applied");
		}
		
		return 2;
	}
	
	
	
	public static boolean isHashing(Properties prop)
	{
		return Boolean.parseBoolean(prop.getProperty("encoding.hash")); 
	}
	
	
	
	
	public static boolean allow4Window(Properties prop)
	{
		boolean out = true;
		try
		{
			out = Boolean.parseBoolean(prop.getProperty("mlp.windows"));
		}
		catch(Exception e) {}
		return out; 
	}
	
	public static boolean includesBias(Properties prop)
	{
		return Boolean.parseBoolean(prop.getProperty("mlp.bias")); 
	}
	
	public static boolean includesRandomClients(Properties prop)
	{
		return Boolean.parseBoolean(prop.getProperty("encoding.random.clients")); 
	}
	
	public static int getChannels(Properties prop)
	{
		return Integer.parseInt(prop.getProperty("encoding.channels"));
	}
	
	
	public static double getMin(Properties prop)
	{
		return Double.parseDouble(prop.getProperty("encoding.min"));
	}
	
	public static double getGclipping(Properties prop)
	{
		return Double.parseDouble(prop.getProperty("gradient.clipping"));
	}
	
	
	public static double getClip(Properties prop)
	{
		
		double out = 0;
		try
		{
			out = Double.parseDouble(prop.getProperty("mlp.clip"));
		}
		catch(Exception e) {}
		return out; 
		
	}
	
	
	public static double getBatchSize(Properties prop)
	{
		
		double out = 1;
		try
		{
			out = Double.parseDouble(prop.getProperty("mlp.batch"));
		}
		catch(Exception e) {}
		return out; 
		
	}
	
	
	public static double getMax(Properties prop)
	{
		return Double.parseDouble(prop.getProperty("encoding.max"));
	}
	
	
	public static double getLR(Properties prop)
	{
		return Double.parseDouble(prop.getProperty("learning.rate"));
	}
	
	public static int getNodes(Properties prop)
	{
		return Integer.parseInt(prop.getProperty("mlp.nodes"));
	}
	
	public static int getEpochs(Properties prop)
	{
		return Integer.parseInt(prop.getProperty("mlp.epochs"));
	}
	
	public static int getIterations(Properties prop)
	{
		return Integer.parseInt(prop.getProperty("mlp.iterations"));
	}
	
	
	public static int[] getLayers(Properties prop)
	{
		String layers = prop.getProperty("mlp.layers");
		String[] parts = layers.split(",");
		int[] output = new int[parts.length];
		
		for(int i=0; i<parts.length; i++)
		{
			output[i] = Integer.parseInt(parts[i]);
		}
		
		return output;
	}
	
	public static ActivationFunction.Types getActivationFunc(Properties prop)
	{
		String af = prop.getProperty("mlp.af");
		
		ActivationFunction.Types result = null;
		
		    switch (af) {
		        case "sigmoid":
		            result = ActivationFunction.Types.SIGMOID; 
		            break;
		        case "relu":
		            result = ActivationFunction.Types.ReLU;
		            break;
		        case "tanh":
		            result = ActivationFunction.Types.TANH;
		            break;
		       
		    }
		    return result;
	}
	
	
	public static NoiseGenerator getNoiseGenerator(Properties prop)
	{
		String af = prop.getProperty("noise.generator");
		
		NoiseGenerator generator = null;
		
		if(af==null) return null;
		
		switch (af) {
		case "gaussian":
			generator = new GaussianNoise(); 
			break;
		case "laplace":
			generator = null;
			break;

		}
		    return generator;
	}
	

	public static void dumpEnc2File(ArrayList<HashMap<String, EncodedWeight[][]>> allWeights, String filename)
	{
		 try
         {
			 SimpleDateFormat formatter = new SimpleDateFormat("dd.MM.yy_HH_mm_ss");  
             Date date = new Date();
             filename = formatter.format(date)+"_"+filename;
             System.out.println(filename);  
         	 FileOutputStream fos =
                   new FileOutputStream(filename);
             ObjectOutputStream oos = new ObjectOutputStream(fos);
             
             if(allWeights.size()>=2)
             {
            	 HashMap<String, EncodedWeight[][]> bias = allWeights.get(1);
            	 if((bias==null)||(bias.size()==0))
            		 allWeights.remove(1);
             }
             
             oos.writeObject(allWeights);
             oos.close();
             fos.close();                    
             System.out.printf("Serialized HashMap size "+allWeights.size()+" data is saved in "+filename);
         }catch(IOException ioe)
          {
                ioe.printStackTrace();
          }
	}
	
	public static void dump2File(ArrayList<HashMap<String, double[][]>> allWeights, String filename)
	{
		 try
         {
			 SimpleDateFormat formatter = new SimpleDateFormat("dd.MM.yy_HH_mm_ss");  
             Date date = new Date();
             filename = formatter.format(date)+"_"+filename;
             System.out.println(filename);  
         	 FileOutputStream fos =
                   new FileOutputStream(filename);
             ObjectOutputStream oos = new ObjectOutputStream(fos);
             
             if(allWeights.size()>=2)
             {
            	 HashMap<String, double[][]> bias = allWeights.get(1);
            	 if((bias==null)||(bias.size()==0))
            		 allWeights.remove(1);
             }
             
             oos.writeObject(allWeights);
             oos.close();
             fos.close();                    
             System.out.printf("Serialized HashMap size "+allWeights.get(0).size()+" data is saved in "+filename);
         }catch(IOException ioe)
          {
                ioe.printStackTrace();
          }
	}
	
}

