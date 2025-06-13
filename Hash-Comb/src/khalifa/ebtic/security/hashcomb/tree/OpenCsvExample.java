package khalifa.ebtic.security.hashcomb.tree;


import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class OpenCsvExample {

    public static void main(String[] args) throws IOException, CsvException {

        String fileName = "etc\\test.csv";
        try (CSVReader reader = new CSVReader(new FileReader(fileName))) {
            List<String[]> r = reader.readAll();
            
            
            
            r.forEach(x -> System.out.println(Arrays.toString(x)));
        }

    }

}