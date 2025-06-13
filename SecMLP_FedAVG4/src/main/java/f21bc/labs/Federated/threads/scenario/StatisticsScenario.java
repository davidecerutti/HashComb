package f21bc.labs.Federated.threads.scenario;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.Vector;

import f21bc.labs.Federated.Utils;

public class StatisticsScenario {
	
	
	private static Vector<String> getPaths(File[] listOfFiles) throws IOException
	{
		
		int[] index = {5,9};
		
		Vector<String> list = new Vector<String>();
		List<File> datas
        = new ArrayList<File>();

		int i=0;
		for (File f : listOfFiles) {
		    if (f.isFile()) {
		        
		    	String out = "Name: "+f.getCanonicalPath();
		    	
		    	if(i>=index[0] && i<=index[1])
		    	{
		    		list.add(f.getCanonicalPath());
		    		out = out+" :: added";
		    	}
		    	else
		    		out = out+" :: skipped";
		    	
		    	System.out.println(out);
		    	i++;
		    	
		    }
		}
		
		return list;
	}
	

	
//	private static Vector<String> getPaths(File[] listOfFiles) throws IOException
//	{
//		Vector<String> list = new Vector<String>();
//		List<File> datas
//        = new ArrayList<File>();
//
//		int i=0;
//		for (File f : listOfFiles) {
//		    if (f.isFile()) {
//		        
//		    	System.out.println("Name: "+f.getCanonicalPath());
//		    	list.add(f.getCanonicalPath());
//		    	i++;
//		    	
//		    }
//		}
//		
//		return list;
//	}

	
	private static void copyFileUsingChannel(File source, File dest) throws IOException {
	    FileChannel sourceChannel = null;
	    FileChannel destChannel = null;
	    try {
	        sourceChannel = new FileInputStream(source).getChannel();
	        destChannel = new FileOutputStream(dest).getChannel();
	        destChannel.transferFrom(sourceChannel, 0, sourceChannel.size());
	       }finally{
	           sourceChannel.close();
	           destChannel.close();
	   }
	}
	
	
	
	
	
	 public synchronized static void runSimulation(String tempPath) throws IOException, ClassNotFoundException, InterruptedException 
	 {
		 File source = new File(tempPath);	 
		 String currentPath = new java.io.File(".").getCanonicalPath();
		 
		 Path validation_path = Paths.get(currentPath, "data", "HFL", "validation", "validation.csv");
		 
		
		 System.out.println("current file: "+tempPath);
		 
		 File dest = new File(validation_path.toString());
		 
		 copyFileUsingChannel(source, dest);
		 
		 
		 boolean isDeleated = false;
		 
		 while(!isDeleated)
		 {
			 isDeleated = source.delete();
			 System.out.println("File "+source.getAbsolutePath()+" has been removed: "+isDeleated);
			 Thread.sleep(6000);
		 }
		 
		 
		 ServerThread thread = new ServerThread();
		 
		thread.start();
		
		System.gc();
		Thread.sleep(10000);
	
		 
		 
		 
		 ClientManager2.main(null);
		 
		 copyFileUsingChannel(dest, source);
		 
		 dest.delete();
		 
	 }
	
	public static void main(String[] args) throws ClassNotFoundException, IOException, InterruptedException
	{
		
		String subDir = "HFL";
		String dir = "data"+File.separator+subDir+File.separator;
		
		String propFile = "configuration.prop";
		Properties prop = Utils.loadConf(propFile);
		
		String file = Utils.getDataFile(prop);
		
		File folder = new File(dir);
		File[] listOfFiles = folder.listFiles();
		
		
		Vector<String> paths = getPaths(listOfFiles);
		
		
		Iterator<String> iter = paths.iterator();
		while(iter.hasNext())
		{
			runSimulation(iter.next());
		}
		
		
	}

}
