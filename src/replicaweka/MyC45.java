/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package replicaweka;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.BinC45ModelSelection;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Enumeration;
import java.util.Vector;

public class MyC45 extends Classifier {

  private ClassifierTree m_root;
  
  private float Confidence = 0.01f;
  
  private boolean is_pruned = true;

  private int minimalAttr = 2;

  private boolean moveSubtree = true;

  private boolean cleanMemory = false;
  

  
  public void createModel(Instances instances) 
       throws Exception {

    ModelSelection modSelection;	 

    modSelection = new C45ModelSelection(minimalAttr, instances);
    m_root = new C45PruneableClassifierTree(modSelection, is_pruned, Confidence,moveSubtree, !cleanMemory);
    m_root.buildClassifier(instances);
    ((C45ModelSelection)modSelection).cleanup();
  }


  public String toString() {

    if (m_root == null) {
      return "No classifier built";
    }
    if (!is_pruned)
      return "MyC45 unpruned tree\n------------------\n" + m_root.toString();
    else
      return "MyC45 pruned tree\n------------------\n" + m_root.toString();
  }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        createModel(i);
    }
    
    public double classifyInstance(Instance instance) throws Exception {
        return m_root.classifyInstance(instance);
    }

  

}
