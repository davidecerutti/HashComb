package ebtic.labs.utils;

import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.logging.Level;

import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.WindowConstants;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import ebtic.labs.metrics.Accuracy;
import ebtic.labs.metrics.History;

public class PlotTraining extends ApplicationFrame {

/**
 * A demonstration application showing an XY series containing a null value.
 *
 * @param title  the frame title.
 */
	
	public PlotTraining(final String title, History history, String content) {

	    super(title);
	    final XYSeries series = new XYSeries("Accuracy");
	    double[][] labels=history.getTrainingLabels();
	    int counter = history.getEpochs();
	     
	    for(int i=0; i<counter; i++)
	    {
	    	Accuracy acc = history.getAccuracy(i);
	    	double[] out = acc.getValue(labels, 1000);
	    	series.add(i, out[1]);
	    }
	    
	    final XYSeriesCollection data = new XYSeriesCollection(series);
	    final JFreeChart chart = ChartFactory.createXYLineChart(
	        title,
	        "epochs", 
	        "acc", 
	        data,
	        PlotOrientation.VERTICAL,
	        true,
	        true,
	        false
	    );
	    
	    final JPanel panel = new JPanel();
	    panel.setLayout(new GridLayout(0, 1));

	    final ChartPanel chartPanel = new ChartPanel(chart);
	    chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
//	    setContentPane(chartPanel);
	    final JTextArea txtOutput = new JTextArea(20,50);
	    txtOutput.setFont( new Font("monospaced", Font.PLAIN, 10) );
	    txtOutput.append(content);
//	    setContentPane(txtOutput);
	    panel.add(chartPanel);
	    panel.add(txtOutput);
	    setContentPane(panel);
	    
	}

	
	   /**
     * Listens for the main window closing, and shuts down the application.
     *
     * @param event  information about the window event.
     */
    public void windowClosing(final WindowEvent event) {
        if (event.getWindow() == this) {
            dispose();
//            System.exit(0);
        }
    }
	
	
// the code has been taken and adapted for our purpose from the Demo samples of the JFreeChart project, link below: 	
// ****************************************************************************
// * JFREECHART DEVELOPER GUIDE                                               *
// * The JFreeChart Developer Guide, written by David Gilbert, is available   *
// * to purchase from Object Refinery Limited:                                *
// *                                                                          *
// * http://www.object-refinery.com/jfreechart/guide.html                     *
// *                                                                          *
// * Sales are used to provide funding for the JFreeChart project - please    * 
// * support us so that we can continue developing free software.             *
// ****************************************************************************
/**
 * Starting point for the demonstration application.
 *
 * @param args  ignored.
 */
public static void start(final String title, History history, String content, boolean allowed) {

	if(allowed)
	{
		try 
		{
			final PlotTraining demo = new PlotTraining(title, history, content);
		    demo.pack();
		    demo.setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE);
		    RefineryUtilities.centerFrameOnScreen(demo);
		    demo.setVisible(true);
		    //demo.setVisible(false);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	

    
}

}