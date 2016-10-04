/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package replicaweka;

import weka.core.Instances;

/**
 *
 * @author Satria, Cliff, Luqman
 */
public class MyID3 {
    public void createModel(Instances trainDataset){
        System.out.println("Creating model for ID3");
    }
    
    public void testDataset(Instances trainDataset){
        System.out.println("Test Dataset model for ID3");
    }
    
    public void crossFold(Instances trainDataset){
        System.out.println("Cross Fold model for ID3");
    }
    
    public void percentageSplit(Instances trainDataset){
        System.out.println("Percentage Split model for ID3");
    }
    
    public void predict(Instances trainDataset){
        System.out.println("Predict model for ID3");
    }
    
}
