/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package replicaweka;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/**
 *
 * @author Satria, Cliff, Luqman
 */
public class ID3 extends Classifier {
    private ID3[] m_Successors;
    
    private Attribute m_Attribute;
    private Attribute m_ClassAttribute;
    
    private double m_ClassValue;
    private double[] m_Distribution;
       
    
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        result.setMinimumNumberInstances(0);

        return result;
      }

    @Override
    public void buildClassifier(Instances instance) throws Exception {
        getCapabilities().testWithFail(instance);      // test if classifier can handle the data
        
        instance = new Instances(instance);
        instance.deleteWithMissingClass();
        
        makeTree(instance);
    }

    private void makeTree(Instances instance) throws Exception {
        if (instance.numInstances() == 0) {
            m_Attribute = null;
            m_ClassValue = Instance.missingValue();
            m_Distribution = new double[instance.numClasses()];
            return;
        }

        double[] infoGains = new double[instance.numAttributes()];
        Enumeration attEnum = instance.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGains[att.index()] = computeIG(instance, att);
        }
        m_Attribute = instance.attribute(Utils.maxIndex(infoGains));

        if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
            m_Attribute = null;
            m_Distribution = new double[instance.numClasses()];
            Enumeration instEnum = instance.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                m_Distribution[(int) inst.classValue()]++;
            }
            Utils.normalize(m_Distribution);
            m_ClassValue = Utils.maxIndex(m_Distribution);
            m_ClassAttribute = instance.classAttribute();
        } else {
            Instances[] splitData = splitData(instance, m_Attribute);
            m_Successors = new ID3[m_Attribute.numValues()];
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                m_Successors[j] = new ID3();
                m_Successors[j].makeTree(splitData[j]);
            }
        }
    }

    private double computeIG(Instances instance, Attribute att) throws Exception {
        double infoGain = computeEntropy(instance);
        Instances[] splitData = splitData(instance, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances() / (double) instance.numInstances()) * computeEntropy(splitData[j]);
            }
        }
        return infoGain;
    }
    
    private double computeEntropy(Instances data) throws Exception {

    double [] classCounts = new double[data.numClasses()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      classCounts[(int) inst.classValue()]++;
    }
    double entropy = 0;
    for (int j = 0; j < data.numClasses(); j++) {
      if (classCounts[j] > 0) {
        entropy -= classCounts[j] * Utils.log2(classCounts[j]);
      }
    }
    entropy /= (double) data.numInstances();
    return entropy + Utils.log2(data.numInstances());
  }

    private Instances[] splitData(Instances instance, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
          splitData[j] = new Instances(instance, instance.numInstances());
        }
        Enumeration instEnum = instance.enumerateInstances();
        while (instEnum.hasMoreElements()) {
          Instance inst = (Instance) instEnum.nextElement();
          splitData[(int) inst.value(att)].add(inst);
        }
        for (int i = 0; i < splitData.length; i++) {
          splitData[i].compactify();
        }
        return splitData;
    }
    
    public double classifyInstance(Instance instance) 
        throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, " + "please.");
        }
        if (m_Attribute == null) {
          return m_ClassValue;
        } else {
            return m_Successors[(int) instance.value(m_Attribute)].
                classifyInstance(instance);
        }
    }
    
    public String toString() {

        if ((m_Distribution == null) && (m_Successors == null)) {
          return "Id3: No model built yet.";
        }
        return "\nId3\n" + printTree(0);
    }
    
    private String printTree(int level) {

        StringBuffer text = new StringBuffer();

        if (m_Attribute == null) {
            if (Instance.isMissingValue(m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
            } 
        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
                text.append(m_Successors[j].printTree(level + 1));
            }
        }
        return text.toString();
      }
}
