package weka.clusterers;

import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.pmml.jaxbbindings.TimeSeries;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

/*
 * 
 * <!-- globalinfo-start --> Cluster data using the FAST algorithm. Can use
 * either the Krushkal's algorithm (default) or the Prim's algorithm. In 
 * this algorithm we have used Krushkal's algorithm:<br/>
 * 
 * <!-- globalinfo-end -->
 * 
 * @author Smit Shilu (smitshilu@gmail.com) 
 * @author Kushal Sheth (kushalsheth28@gmail.com)
 * @version $Revision: 1 $
 * 
 * 
 */

public class FASTAlgorithm extends AbstractFASTClusterer  {
	
	Instances m_instances;
        List<Integer> cluster;
        String[] m_ClusterName;
        double execution_time;
        
	public FASTAlgorithm(){
		super();
	}

        @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();
    result.enable(Capabilities.Capability.NO_CLASS);

    
    // attributes
    result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capabilities.Capability.MISSING_VALUES);

    return result;
  }
        
        @Override
	public void buildClusterer(Instances data) throws Exception {


		/**
		 * get every instances from dataset
		 */
		//m_instances = data;
		//int nInstances = m_instances.numInstances();

		/**
		 * set class index
		 */
		data.setClassIndex(data.numAttributes() - 1);

		/**
		 * Filter numeric data in nominal to calculate entropy
		 */
		NumericToNominal ntm = new NumericToNominal();
		ntm.setInputFormat(data);
		data = Filter.useFilter(data, ntm);

		/**
		 * call InfoGainAttributeEval to calculate entropy
		 */
		AttributeInfoEval aiv = new AttributeInfoEval();
		aiv.buildEvaluator(data);

		/**
		 * get Trelevance
		 */
		double m_trelevance[] = aiv.getTRelevance();

		RemoveAttributes ra = new RemoveAttributes();
		data = ra.RemoveIrrelevantAttribute(data, m_trelevance);

		aiv.buildEvaluator(data);
		m_trelevance = aiv.getTRelevance();

		double[][] m_correlation = aiv.getCorrelation(data);

		KruskalAlgorithm ka = new KruskalAlgorithm(m_correlation.length);
		double m_mst[][] = ka.kruskalAlgorithm(m_correlation);

		ClusterGeneration cg = new ClusterGeneration();
		double m_forest[][] = cg.ForestGeneration(m_mst, m_trelevance);

		cluster = cg.getForest(m_forest, m_trelevance);
                
                m_ClusterName = new String[cluster.size()];
                
		for (int i = 0; i < cluster.size(); i++) {                        
                      m_ClusterName[i] = data.attribute(cluster.get(i)).name();
		}

	}

  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   */

	
	  public static void main(String[] argv) {
		    runClusterer(new FASTAlgorithm(), argv);
                    //System.out.println(m_ClusterName.toString);
		  }

    @Override
    public int numberOfClusters() throws Exception {
        return cluster.size();
    }	
    
    public String toString(boolean header) {
    StringBuffer temp = new StringBuffer();

    if (m_ClusterName == null) {
      return "No clusterer built yet";
    }

    if (header) {
      temp.append("\nFAST clustering\n=================\n");
      temp.append("\nNumber of attributes selected: "
        + m_ClusterName.length);
    }

    temp.append("\n\n");

    for(int i = 0 ; i < m_ClusterName.length ; i++)
    temp.append(m_ClusterName[i]+"\n");

    temp.append("\n");

    return temp.toString();
  }
	

}
