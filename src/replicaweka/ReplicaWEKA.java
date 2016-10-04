/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
`*Bagian 1: menuliskan kode java untuk mengakses weka, 
•mulai dari load data(arrf dan csv) done
•remove atribut done
•Filter : Resample done
•build classifier : J48, DT done
•testing model given test set, done
•10-fold cross validation, done
•percentage split, done
•Save/Load Model, done
•using model to classify one unseen data (input data)
  *Bagian2:  membuat Classifier baru dengan menurunkan dari Classifier WEKA
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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.Stream;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Satria, Cliff, Luqman
 */
public class ReplicaWEKA {
    private static boolean isLoad;
    private static Classifier loadedClassifier;
    private static String path = "C:\\Users\\Satria\\Documents\\NetBeansProjects\\replicaWEKA\\weka_models\\";
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        System.out.println("Enter Type (weka/replicaWEKA/saveLoad)");
        Scanner sc = new Scanner(System.in);
        String input = sc.nextLine();
        if (input.equals("weka")){
            classifyWeka();
        } else if (input.equals("replicaWEKA")){
            readFile("test.arff");
            ID3 id3 = new ID3();
            C45 c45 = new C45();
            System.out.println("Start ID3");
            id3.createModel();
            System.out.println("Finish ID3");

            System.out.println("Start C4.5");
            c45.createModel();
            System.out.println("Finish C4.5");
        } else if (input.equals("saveLoad")){
            loadedClassifier = loadModel(new File(path),"w.nom.crossfold");
            isLoad = true;
            classifyWeka();
        } else {
            System.out.println("Not valid input");
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
    // Save and load model are code written by  snake plissken at Mar 6 '14 at 7:49 http://stackoverflow.com/questions/22201949/save-and-load-smo-weka-model-from-file
    private static void saveModel(Classifier c, String name) throws Exception {

        ObjectOutputStream oos = null;
        try {
            oos = new ObjectOutputStream(
                    new FileOutputStream(path + "\\" + name + ".model"));

        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
        oos.writeObject(c);
        oos.flush();
        oos.close();

    }
    
    private static Classifier loadModel(File path, String name) throws Exception {

        Classifier classifier;

        FileInputStream fis = new FileInputStream(path + "\\"+ name + ".model");
        ObjectInputStream ois = new ObjectInputStream(fis);

        classifier = (Classifier) ois.readObject();
        ois.close();

        return classifier;
    }
    
    //Remove some instance for Spec no.2 but is not currently needed
    public static void filterRemove(Instances trainDataset) throws Exception{
        
        String[] options = new String[2];
        options[0] = "-R";                                              // "range"
        options[1] = "2";                                               // first attribute will be removed
        Remove remove = new Remove();                                   // new instance of filter
        remove.setAttributeIndices("1");    // apply filter
        remove.setInputFormat(trainDataset);
        Instances newTrain = Filter.useFilter(trainDataset, remove);
        for (int i = 0; i < newTrain.numInstances(); i++) {
            Instance inst = newTrain.instance(i);
            System.out.println(inst.toString());
        }
        
        
    }
    
    //Resample some instance for Spec no.3 but is not currently needed
    public static void filterResample(Instances trainDataset) throws Exception{
        
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
        
    }
    
    public static void classifyWeka() throws Exception{
        double accuracy = 0.0f;
        int correctPrediction = 0;
        //Get the train
        DataSource dt = new DataSource("weather.nominal.arff");
        Instances trainDataset = dt.getDataSet();
        //set the index of class
        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
        
        
        if (isLoad){
            System.out.println("Clasify from load model");
            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(loadedClassifier, trainDataset);
            System.out.println(eval.toSummaryString("\nResults Test Dataset from Loaded model\n======\n", false));
            saveModel(loadedClassifier,"saveTio");
            System.out.println("Model has been saved");
            return;
            
        }
        
        //Get the test
        dt = new DataSource("weather.nominal.test.arff");
        Instances testDataset = dt.getDataSet();
        //set the index of class
        testDataset.setClassIndex(testDataset.numAttributes() - 1);
        //read The Classifier
        System.out.println("Enter Classifier (id3/c45)");
        Scanner sc = new Scanner(System.in);
        String inputClassifier = sc.nextLine();
        if(inputClassifier.equals("id3")){
            System.out.println("Building ID3 WEKA");
            Id3 id3 = new Id3();
            id3.buildClassifier(trainDataset);
            System.out.println("Enter Test Type (suppliedtest/10kcrossfold/percentage)");
            sc = new Scanner(System.in);
            String inputTester = sc.nextLine();
            if (inputTester.equals("suppliedtest")){
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
                Evaluation eval = new Evaluation(trainDataset);
                eval.evaluateModel(id3, testDataset);
                System.out.println(eval.toSummaryString("\nResults Test Dataset\n======\n", false));
                
            } else if (inputTester.equals("10kcrossfold")){
                Evaluation eval = new Evaluation(trainDataset);
                eval.crossValidateModel(id3, trainDataset, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nResults 10k Cross Fold Validation\n======\n", false));
            } else if (inputTester.equals("percentage")){
                //Randomize first to make sure no part has incremental data
                trainDataset.randomize(new java.util.Random(0));
                //split the size based on percentage
                int percent = 80;
                int trainSize = (int) Math.round(trainDataset.numInstances() * percent
                    / 100);
                int testSize = trainDataset.numInstances() - trainSize;
                Instances train = new Instances(trainDataset, 0, trainSize);
                Instances test = new Instances(trainDataset, trainSize, testSize);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(id3, test);
                System.out.println(eval.toSummaryString("\nResults Percentage Split\n======\n", false));
            } else {
                System.out.println("Not valid test Type");
            }
            
        } else if(inputClassifier.equals("c45")){
            System.out.println("Building C45/J48 WEKA");
            J48 j48 = new J48();
            j48.buildClassifier(trainDataset);System.out.println("Enter Test Type (suppliedtest/10kcrossfold/percentage)");
            sc = new Scanner(System.in);
            String inputTester = sc.nextLine();
            if (inputTester.equals("suppliedtest")){
                //Compare result
                for (int i = 0; i < testDataset.numInstances(); i++) {
                    double realClass = testDataset.instance(i).classValue();
                    String realClassName = testDataset.classAttribute().value((int) realClass);
                    //Predict the instance
                    Instance inst = testDataset.instance(i);
                    double predictionClass = j48.classifyInstance(inst);
                    String predictionClassName = testDataset.classAttribute().value((int) predictionClass);
                    System.out.println("R: " + realClassName + " , P: " + predictionClassName );
                    if (realClassName.equals(predictionClassName)){
                        correctPrediction++;
                    }
                }
                System.out.println("Correct pred : " + correctPrediction);
                accuracy = correctPrediction / testDataset.numInstances();
                System.out.println("Accuracy : " + accuracy);
                Evaluation eval = new Evaluation(trainDataset);
                eval.evaluateModel(j48, testDataset);
                System.out.println(eval.toSummaryString("\nResults Test Dataset\n======\n", false));
                
            } else if (inputTester.equals("10kcrossfold")){
                Evaluation eval = new Evaluation(trainDataset);
                eval.crossValidateModel(j48, trainDataset, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nResults 10k Cross Fold Validation\n======\n", false));
            } else if (inputTester.equals("percentage")){
                //Randomize first to make sure no part has incremental data
                trainDataset.randomize(new java.util.Random(0));
                //split the size based on percentage
                int percent = 80;
                int trainSize = (int) Math.round(trainDataset.numInstances() * percent
                    / 100);
                int testSize = trainDataset.numInstances() - trainSize;
                Instances train = new Instances(trainDataset, 0, trainSize);
                Instances test = new Instances(trainDataset, trainSize, testSize);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(j48, test);
                System.out.println(eval.toSummaryString("\nResults Percentage Split\n======\n", false));
            } else {
                System.out.println("Not valid test Type");
            }
        }
        
    }
}
