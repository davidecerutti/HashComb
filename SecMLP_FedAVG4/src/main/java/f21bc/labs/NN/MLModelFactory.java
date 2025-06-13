package f21bc.labs.NN;

import java.lang.reflect.Constructor;

public class MLModelFactory {
	
	 public static Class<?> getPrimitiveType(Class<?> clazz) {
	        if (clazz == Integer.class) return int.class;
	        if (clazz == Double.class) return double.class;
	        if (clazz == Boolean.class) return boolean.class;
	        if (clazz == Float.class) return float.class;
	        if (clazz == Long.class) return long.class;
	        if (clazz == Short.class) return short.class;
	        if (clazz == Byte.class) return byte.class;
	        if (clazz == Character.class) return char.class;
	        return clazz;  // Return original class if it's not a wrapper type
	    }
	

    // Dynamically load the class and create an instance
    public static NN getModelType(String className, Object... constructorArgs) {
        try {
            // Load the class by name
            Class<?> clazz = Class.forName(className);
            
            // Get parameter types from constructorArgs (infer types for reflection)
            Class<?>[] paramTypes = new Class<?>[constructorArgs.length];
            for (int i = 0; i < constructorArgs.length; i++) {
            	paramTypes[i] = getPrimitiveType(constructorArgs[i].getClass());
            }

            // Retrieve the appropriate constructor
            Constructor<?> constructor = clazz.getDeclaredConstructor(paramTypes);

            // Create a new instance by passing arguments to the constructor
            return (NN) constructor.newInstance(constructorArgs);
            
        } catch (ClassNotFoundException e) {
            System.out.println("Class not found: " + className);
        } catch (Exception e) {
        	e.printStackTrace();
            System.out.println("Error creating an instance of: " + className);
        }
        
        return null; // Return null if class loading or instantiation fails
    }


}
