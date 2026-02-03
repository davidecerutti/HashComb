package ebtic.labs.tests;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.Exceptions.ConfigurationException;
import ebtic.labs.Exceptions.MatrixException;
import ebtic.labs.LF.LossFunction;
import ebtic.labs.NN.EarlyStopping;
import ebtic.labs.NN.MLP;
import ebtic.labs.metrics.Accuracy;
import ebtic.labs.metrics.History;
import ebtic.labs.metrics.Precision;
import ebtic.labs.metrics.Recall;
import ebtic.labs.objects.Generic;
import ebtic.labs.objects.MyObject;
import ebtic.labs.utils.Matrix;
import ebtic.labs.utils.PlotTraining;
import ebtic.labs.utils.Util;
import ebtic.labs.utils.WeightsExtractor;

/**
 * SingleRun_test is a main class to run the model over training data
 *
 */
public class SingleRun_test {

	
	public static void main(String[] args) throws IOException, MatrixException {
		
		
		//String filenameString = "spam_hf_4.P.csv";
		//String filenameString = "spam_hf_6_original.P.csv";
		String filenameString = "spam_hf_wf_6.P.csv";
		
		
		String subDir = "HFL";
		//String subDir = "VFL";
		
		
		
		//1:Retrieve dataset from CSV File  
        //String fileName = "data\\random_data_banknote_authentication.csv"; 
        //String fileName = "data\\biased_data_banknote_authentication.csv"; 
        //String fileName = "data\\coronary_disease.csv";
        String fileName = "data"+File.separator+subDir+File.separator+filenameString;
		//String fileName = "data\\data_banknote_authentication.csv"; 
        //String fileName = "data\\processed_data.csv"; 
        List<MyObject> beans = Generic.instanciateCSV(fileName);
        
        beans.forEach(System.out::println);
        System.out.println(beans.size());
//        for(int i=0; i<beans.size(); i++)
//        {
//        	Generic d =(Generic) beans.get(i);
//        	double[] data = d.returnData();
//        	String out = "";
//        	for(int j=0; j<data.length; j++)
//        		out = out + data[j]+" ";
//        	System.out.println(i+" Data: "+out);
//        	System.out.println(i+" Label: "+d.returnLabel());
//        	System.out.println();
//        }
        //2:Shuffle the dataset
        Collections.shuffle(beans);
       //3:get the train data and labels
        double[][] m = Matrix.getData(beans);
        double[][] labels = Matrix.getLabels(beans);
	
      //4:set the number of inner layers and number of node in each layer
       
//        int[] layers= {10, 5};        
//      int[] layers= {25, 10, 5};
//      int[] layers= {50, 25, 25, 50};
      int[] layers= {50, 25, 20, 25, 50};
//      int[] layers= {50, 35, 30, 25, 20};
//      int[] layers= {50, 25, 10, 5};
//      int[] layers= {150, 100, 50};
        
        MLP mlp;
		try 
		{
			long startTime = System.nanoTime();
			//5:determine the activation function type
			ActivationFunction.Types af = ActivationFunction.Types.SIGMOID;
			
			//6:determine the early stop
			EarlyStopping earlyStop = new EarlyStopping(0.0015, 20);
//			EarlyStopping earlyStop = new EarlyStopping(0.00015, 20);
			
			//7:Initialize the Model with hyperparameters
			int epochs = 30000;
			mlp = new MLP(epochs, 0.05, layers, af, LossFunction.Types.CE ,earlyStop,2);
			
			
			
			
			
			//8:fit the training data
//			History history = mlp.fit(m, labels);
			
			// also allows to set the batch mode (in range R between [0, 1])
			WeightsExtractor extractor = mlp.getWeightExtractor();
			
			// this is to set the first epoch to start writing the weights on file, the default is the beginning of the training
			extractor.setStop(5000);
			//extractor.setFileName("biased4.banknote.csv", 4, true);
			extractor.setFileName(filenameString, (layers.length - 1), true);
			//extractor.setFileName("biased6.banknote.csv", 4, true);
			History history = mlp.fit(m, labels, 0.1);
			
			
			
			double[][] x_test = m;
			double[][] y_test = labels;
			
			//9:get the Metric data.
			Accuracy acc = history.getAccuracy(History.Type.TRAINING);
			Precision prec = history.getPrecision(History.Type.TRAINING);
			Recall rec = history.getRecall(History.Type.TRAINING);
			
			
			try
			{
				//10:show the training result
				String printing1 = Util.metricsTable(prec, rec, acc, y_test, af);
				System.out.println(printing1);
				long endTime = System.nanoTime();
				long timeElapsed = endTime - startTime;
				 
		        //System.out.println("Execution time in nanoseconds: " + timeElapsed);
		        System.out.println("Execution time in milliseconds: " + timeElapsed / 1000000);
				
		        PlotTraining.start("Training:", history, printing1, true);


				
			}
			catch(NullPointerException e) {e.printStackTrace();}
			
			
//			//10:show the training result
//			String printing1 = Util.metricsTable(prec, rec, acc, y_test, af);
//			System.out.println(printing1);
//			long endTime = System.nanoTime();
//			long timeElapsed = endTime - startTime;
//			 
//	        //System.out.println("Execution time in nanoseconds: " + timeElapsed);
//	        System.out.println("Execution time in milliseconds: " + timeElapsed / 1000000);		
//	        PlotTraining.start("Training:", history, printing1, true);
			
		} catch (ConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        
        
	}
	
	
}
