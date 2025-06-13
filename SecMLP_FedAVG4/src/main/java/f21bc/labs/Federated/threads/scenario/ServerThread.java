package f21bc.labs.Federated.threads.scenario;

import java.io.IOException;

public class ServerThread extends Thread {
	
	
	
	  public static void main(String[] args) {
		ServerThread thread = new ServerThread();
	    thread.start();
	    System.out.println("This code is outside of the thread");
	  }
//	  public void run() {
//		
//		boolean check = true;
//		try {
//			Thread.sleep(1000);
//		} catch (InterruptedException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//		}
//		while(check)
//		{
//			try 
//			{
//				check = false;  
//				ServerManager.main(null);
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				//e.printStackTrace();
//				System.out.println("Server Address still in use!!!!!!!!!");
//				check = true;
//				
//			}
//			
//		}
//
//	  }
	  
	  
	  public void run() {
		  
		  boolean repeat = true;
		  while(repeat)
		  {
			  try 
				{
					System.out.println("The server is starting!!!");
					
					repeat = false;
					ServerManager.main(null);
					System.out.println("The server has been terminated!!!");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					//e.printStackTrace();
					System.out.println("The server is still shutting down, not ready... need to wait!!!");
					repeat = true;
					try {
						Thread.sleep(15000);
					} catch (InterruptedException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}
				}    
		  }
		
	  }
	  
	  
	}