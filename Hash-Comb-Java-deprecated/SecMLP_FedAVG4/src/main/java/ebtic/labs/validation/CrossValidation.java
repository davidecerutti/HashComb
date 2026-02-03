package ebtic.labs.validation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import ebtic.labs.AF.ActivationFunction;
import ebtic.labs.Exceptions.CrossValidationException;
import ebtic.labs.Exceptions.MatrixException;
import ebtic.labs.NN.NN;
import ebtic.labs.metrics.Accuracy;
import ebtic.labs.metrics.History;
import ebtic.labs.metrics.Precision;
import ebtic.labs.metrics.Recall;
import ebtic.labs.utils.Matrix;
import ebtic.labs.utils.PlotTraining;
import ebtic.labs.utils.Util;

public class CrossValidation {
	
	
	
	public CrossValidation()
	{
		
	}
	
	private static double[][] getWithinFrame(double[][] matrix, int index, int last)
	{
		if(last>=matrix.length)
			last = matrix.length;
		return Matrix.subMatrix(matrix, index, last);
	}
	
	
	private static double[][] getOutsideFrame(double[][] matrix, int index, int last)
	{
		int rows = matrix.length;
		
		// the fold is the first of the samples
		if(index == 0)
			return Matrix.subMatrix(matrix, last, rows-1);
		
		// the fold is the last of the samples
		else if(last>=matrix.length)
		{
			return Matrix.subMatrix(matrix, 0, index-1);
		}
		
		// the fold is in the middle
		else
		{
			double[][] a = Matrix.subMatrix(matrix, 0, index-1);
			double[][] b = Matrix.subMatrix(matrix, last, rows-1);
			return Matrix.concatMatrix(a, b);
		}
		
	}
	
	
	public static ArrayList<History> cross_validate(NN model, double[][] X, double[][] Y, int folds, ActivationFunction.Types af) throws CrossValidationException, IOException
	{
		String error = "Wrong K-Fold setting for the input";
		int rows = X.length;
		
		ArrayList<History> CV_hystory = new ArrayList<History>();
//		ArrayList<Accuracy> validation_acc = new ArrayList<Accuracy>();
		
		int Fsize = rows / folds;
		
		if(Fsize<100)
			throw new CrossValidationException(error);
		
		else
		{
			int index=0;
			int last=0;
			int count = 1;
			while((index<rows)&&(count<=folds))
			{
				
				last = index+Fsize;
				if(last<rows)
				{
					if(count==folds)
						last = rows;
					System.out.println("Cross Validation------ iteration: "+count);
					double[][] X_train = getOutsideFrame(X, index, last);
					double[][] X_test = getWithinFrame(X, index, last);
					
					double[][] yT = Matrix.transpose(Y);
					double[][] y_train = Matrix.transpose(getOutsideFrame(yT, index, last));
					double[][] y_test = Matrix.transpose(getWithinFrame(yT, index, last));
				
					index=last; 
					
					try
					{
						History history = model.fit(X_train, y_train);
						double[][] predictions = model.predict(X_test);
						history.addValidation(predictions);
						
						
						
						
						
						Accuracy acc = history.getAccuracy(History.Type.TRAINING);
						Precision prec = history.getPrecision(History.Type.TRAINING);
						Recall rec = history.getRecall(History.Type.TRAINING);
						
//						String printing = acc.printAccuracy(y_train); 
//						System.out.println("Training: "+count+"\n"+printing);
						
						String printing1 = Util.metricsTable(prec, rec, acc, y_train, af);
//						System.out.println("Training: "+count+"\n"+printing1);
						
//						acc = history.getAccuracy(History.Type.VALIDATION);
						
						acc = history.getAccuracy(History.Type.VALIDATION);
						prec = history.getPrecision(History.Type.VALIDATION);
						rec = history.getRecall(History.Type.VALIDATION);

						String printing2 = Util.metricsTable(prec, rec, acc, y_test, af);
//						printing = acc.printAccuracy(y_test); 
//						System.out.println("Validation: "+count+"\n"+printing2);
						
						printing1="Training: "+count+"\n"+printing1+"\n"+"Validation: "+count+"\n"+printing2	;
						System.out.println(printing1);
						
						

						history.setTrainingLabels(y_train);
						history.setTestingLabels(y_test);
						
						PlotTraining.start("Training "+count, history, printing1, true);
						
						
						
						
						CV_hystory.add(history);
						
						
					} catch (MatrixException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					count++;
				}
				
			}
			
		}
		return CV_hystory;
		
	}
	

	
	
	
}
