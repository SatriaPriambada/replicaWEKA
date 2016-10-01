/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package replicaweka;

/**
 *
 * @author Satria
 */
public class ReplicaWEKA {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
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
