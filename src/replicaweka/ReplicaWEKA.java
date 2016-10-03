/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
`*Bagian 1: menuliskan kode java untuk mengakses weka, 
•mulai dari load data(arrf dan csv) done
•remove atribut done
•Filter : Resample done
•build classifier : J48, DT
•testing model given test set, 
•10-fold cross validation, percentage split, 
•Save/Load Model,
•using model to classify one unseen data (input data)
•Bagian2:  membuat Classifier baru dengan menurunkan dari Classifier WEKA
•Implementasi kelas baru pada weka: myID3,  myC45
•Penanganan binary class dan multi class
•Penanganan atribut diskrit dan kontinu
Test data yang digunakan: 
-Data binary categorization weather (nominal, kontinu)
-Data multiclass categorization iris
LAPORAN !!! 
> Source Code, Hasil Eksekusi terhadap data tes, perbandingan dengan hasil ID3 &J48 weka(pdf)
 */
package replicaweka;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;
import java.util.stream.Stream;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Satria, Cliff, Luqman
 */
public class ReplicaWEKA {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        String input = sc.nextLine();
        if (input.equals("weka")){
            classifyWeka();
        } else {
            readFile("test.arff");
            ID3 id3 = new ID3();
            C45 c45 = new C45();
            System.out.println("Start ID3");
            id3.createModel();
            System.out.println("Finish ID3");

            System.out.println("Start C4.5");
            c45.createModel();
            System.out.println("Finish C4.5");
        }
    }
    
    public static void readFile(String filename){
        
        //read file into stream, try-with-resources
        try (Stream<String> stream = Files.lines(Paths.get(filename))) {
            stream.forEach(System.out::println);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public static void classifyWeka() throws Exception{
        double accuracy = 0.0f;
        int correctPrediction = 0;
        //Get the train
        DataSource dt = new DataSource("weather.nominal.arff");
        Instances trainDataset = dt.getDataSet();
        //set the index of class
        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        Id3 id3 = new Id3();
        id3.buildClassifier(trainDataset);
        
        //Remove some instance for Spec no.2 but is not currently needed
        /*
        String[] options = new String[2];
        options[0] = "-R";                                              // "range"
        options[1] = "1";                                               // first attribute will be removed
        Remove remove = new Remove();                                   // new instance of filter
        remove.setOptions(options);                                     // set options
        remove.setInputFormat(trainDataset);                            // inform filter about dataset **AFTER** setting options
        Instances newTrain = Filter.useFilter(trainDataset, remove);    // apply filter
        */
        
        //Resample some instance for Spec no.3 but is not currently needed
        /*
        String[] options = new String[6];
        options[0] = "-B";                                              // "bias"
        options[1] = "0.0";                                             // 0 means bias to class index 0 in the case of weather class yes
        options[2] = "-S";                                              // "seed"
        options[3] = "1";                                               // the seed for random Resample
        options[4] = "-Z";                                              // Percentage of Dataset
        options[5] = "100.0";                                           // this mean resample from 100% dataset
        Resample resample = new Resample();                               // new instance of filter
        resample.setOptions(options);                                     // set options
        resample.setInputFormat(trainDataset);                            // inform filter about dataset **AFTER** setting options
        Instances newTrain = Filter.useFilter(trainDataset, resample);    // apply filter
        for (int i = 0; i < newTrain.numInstances(); i++) {
            Instance inst = newTrain.instance(i);
            System.out.println(inst.toString());
            
        }
        */
        //Get the test
        dt = new DataSource("weather.nominal.test.arff");
        Instances testDataset = dt.getDataSet();
        //set the index of class
        testDataset.setClassIndex(testDataset.numAttributes() - 1);
        
        //Compare result
        for (int i = 0; i < testDataset.numInstances(); i++) {
            double realClass = testDataset.instance(i).classValue();
            String realClassName = testDataset.classAttribute().value((int) realClass);
            //Predict the instance
            Instance inst = testDataset.instance(i);
            double predictionClass = id3.classifyInstance(inst);
            String predictionClassName = testDataset.classAttribute().value((int) predictionClass);
            System.out.println("R: " + realClassName + " , " + predictionClassName );
            if (realClassName.equals(predictionClassName)){
                correctPrediction++;
            }
            
        }
        System.out.println("Correct pred : " + correctPrediction);
        accuracy = correctPrediction / testDataset.numInstances();
        System.out.println("Accuracy : " + accuracy);
    }
}
