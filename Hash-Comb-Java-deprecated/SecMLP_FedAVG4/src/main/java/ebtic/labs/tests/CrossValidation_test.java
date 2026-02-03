package ebtic.labs.tests;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.Exceptions.ConfigurationException;
import ebtic.labs.Exceptions.CrossValidationException;
import ebtic.labs.Exceptions.MatrixException;
import ebtic.labs.LF.LossFunction;
import ebtic.labs.NN.EarlyStopping;
import ebtic.labs.NN.MLP;
import ebtic.labs.metrics.History;
import ebtic.labs.objects.Generic;
import ebtic.labs.objects.MyObject;
import ebtic.labs.utils.Matrix;
import ebtic.labs.utils.PlotTraining;
import ebtic.labs.utils.Util;
import ebtic.labs.validation.CrossValidation;
/**
 * CrossValidation_test is a main class to run the model over training and validation data
 *
 */
public class CrossValidation_test {

	
	public static void main(String[] args) throws IOException, MatrixException, CrossValidationException {

		//1:Retrieve dataset from CSV File  
        
		String fileName = "data\\data_banknote_authentication.csv.txt";
        //String fileName = "data\\processed_data.csv";
        List<MyObject> beans = Generic.instanciateCSV(fileName);
         
        System.out.println(beans.size());
        for(int i=0; i<beans.size(); i++)
        {
        	Generic d =(Generic) beans.get(i);
        	double[] data = d.returnData();
        	String out = "";
        	for(int j=0; j<data.length; j++) 
        		out = out + data[j]+" ";
        }
        //2:Shuffle the dataset
        
        Collections.shuffle(beans);
        
        //3:get the train data and labels
        double[][] m = Matrix.getData(beans);
        double[][] labels = Matrix.getLabels(beans);
        
        //4:set the number of inner layers and number of node in each layer
    int[] layers= {10, 8};
//    int[] layers= {25, 10, 5};
//    int[] layers= {50, 25, 25, 50};  
//    int[] layers= {50, 25, 10, 5};
//    int[] layers= {150, 100, 50};

      MLP mlp;
      ArrayList<History> trainingHistory = null;
		try 
		{	
			//5:determine the activation function type
			ActivationFunction.Types af = ActivationFunction.Types.ReLU;
			
			//6:determine the early stop
			EarlyStopping earlyStop = new EarlyStopping(0.09, 20);
			
			//7:Initialize the Model with hyperparameters
			mlp = new MLP(20000, 0.025, layers, af, LossFunction.Types.CE ,earlyStop, 2);
			
			//8:fit and predict the training the validation data
			trainingHistory = CrossValidation.cross_validate(mlp, m, labels, 5, af);			
	
			//9:show the training result
			System.out.println("\n\nAverage results after training: ");
			String output = Util.metricsTable(trainingHistory, History.Type.TRAINING);
			System.out.println(output);
			System.out.println("\n\nAverage results after validating: ");
			output = Util.metricsTable(trainingHistory, History.Type.VALIDATION);
			System.out.println(output);
			
		} catch (ConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
        
	}
	
	
}
