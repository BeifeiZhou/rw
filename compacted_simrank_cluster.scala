package SimRank

import org.apache.spark.rdd._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.graphx._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._
import scala.collection._
import breeze.linalg._
import scala.math._
import scala.util.Random
import scala.util.control.Breaks
import java.util.Collections.EmptyList
import java.util.Collections.EmptyList


object compacted_simrank_cluster {
 def main(args: Array[String]){ 
  class Node(val id: String, val children_ids: List[String], val depth: Int, var parent_id: String) 
  val Conf = new SparkConf().setAppName("simrank")
  Conf.set("spark.hadoop.fs.default.name", "hdfs://master1.hh:8020")
  val sc = new SparkContext(Conf) 
  val data = sc.textFile("hdfs://master1.hh:8020/tmp/sim_rank/albumTags0708.csv")
    .map(s => s.split(","))
    .filter(s => s.length > 1)
    .filter(s => s(1) != "-1")  
  val data_extend = data.map{s=>
      val ss = s.tail
      ss.map(sss => (s(0), sss))
    }.flatMap(s=>s)      
  val data_tag = data_extend.map(s=>(s._2, s._1)).groupBy(s=>s._1).map{s=>
   s._1+","+s._2.map(ss=> ss._2).mkString(",")
  }.map(s=>s.split(",")) 
  
  val topKal = data.map(s => (s(0), s.tail.size)).collect.sortWith((s,t) => s._2 > t._2).map(s => s._1).take(100)
  val topKtag = data_tag.map(s => (s(0), s.tail.size)).collect.sortWith((s,t) => s._2 > t._2).map(s => s._1).take(100)
  
  def rw(numberOfTrees: Int, treeDepth: Int, albumOrTag: String){    
    val topKitem = if (albumOrTag == "album") topKal else topKtag
    val al_tag_dict = if (albumOrTag == "album") data.map(s => (s(0), s.tail)).collect.toMap else data_tag.map(s => (s(0), s.tail)).collect.toMap
    val tag_al_dict = if (albumOrTag == "album") data_tag.map(s => (s(0), s.tail)).collect.toMap else data.map(s => (s(0), s.tail)).collect.toMap  
  
    val n_als = data.map(s=>s(0)).distinct.count
    val n_tags = data_tag.map(s=>s(0)).distinct.count
    
    val als_sc = if (albumOrTag == "album") data.map(s=>s(0)).distinct else data_tag.map(s=>s(0)).distinct
    val als = als_sc.collect
     
    def move(id: String, childrenAll: List[String]) = {
      val tags = al_tag_dict(id)
      val tag = tags(Random.nextInt(tags.length))
      val ancestorsAll = tag_al_dict(tag)
      val ancestors = ancestorsAll diff childrenAll
      if (ancestors.length == 0) "End" else ancestors(Random.nextInt(ancestors.length))
      }
   
      def updateTree(parent_id: String, children_ids: List[String], depth: Int) = {
        val node = new Node(parent_id, children_ids, depth, "")
      ((parent_id, depth), node)
    }
    
    def connect(tree: Map[(String, Int), Node], parent_id: String, children_ids: List[String], depth: Int) = {
      for (child_id <- children_ids){
        tree((child_id, depth-1)).parent_id = parent_id
      }
    }
    
    var trees = List.empty[Map[(String, Int), Node]]
    for (i <- 0 to numberOfTrees-1){
      var tree = als.map(s => updateTree(s, List.empty[String], 0)).toMap
      var step = als_sc.map(s => (move(s, List.empty[String]), s))
        .filter(s => s != "End")
        .groupBy(s => s._1)
        .map(s => (s._1, s._2.toList.map(ss => ss._2)))
      
      var step_coll = step.collect
      step_coll.foreach(s => connect(tree, s._1, s._2, 1))    
      tree = tree ++ step_coll.map(s => updateTree(s._1, s._2, 1)).toMap
  
      for (j <- 2 to treeDepth){
        val stepLoop = step.map(s => (move(s._1, s._2 ++ List(s._1)), s))
          .filter(s => s._1 != "End")
          .groupBy(s => s._1)
          .map(s => (s._1, s._2.toList.map(ss => ss._2)))
        step_coll = stepLoop.map(s => (s._1, s._2.map(ss => ss._1))).collect
        step_coll.foreach(s => connect(tree, s._1, s._2, j))     
        tree = tree ++ step_coll.map(s => updateTree(s._1, s._2, j)).toMap      
        step = stepLoop.map(s => (s._1, s._2.map(ss => ss._1) ++ List.flatten(s._2.map(ss => ss._2))))
      }
      trees = trees ++ List(tree)
    }
    
    def find_all_children_ids(id: String, tree: Map[(String, Int), Node], depth: Int) = {
      val node = tree((id, depth))
      var children = node.children_ids
      if (depth > 1){
        val range = 1 to depth-1
        val seq = range.reverse
        for (i <- seq){
          var newChildren = List.empty[String]
          for (child <- children){
            newChildren = newChildren ++ tree((child, i)).children_ids
          }
          children = newChildren
        }
      }
      children
    }
    
    def query_items(album: String,  trees: List[Map[(String, Int), Node]], topK: Int, topItems: Array[String]) = {
      var sim = Map.empty[String, Double]
      for (tree <- trees){
        var neighbors = List.empty[String]
        val simNeighbor = 0.8
        val node = tree((album, 0))
        val parent_id = node.parent_id
        if (parent_id != ""){
          var ancestor = tree((parent_id,1))
          neighbors = neighbors ++ ancestor.children_ids diff List(album)
          if (neighbors.length != 0){
            for (neighbor <- neighbors){
              if (sim.contains(neighbor)){
                val value = sim(neighbor)+simNeighbor
                sim += neighbor -> value
              }else{
                sim += neighbor -> simNeighbor
              }
            }
          }
          var node = ancestor
          for (i <- 2 to treeDepth){
            val simNeighbor = math.pow(0.8, i)
            var neighbors = List.empty[String]
            var subVertices = List.empty[String]
            val parent_id = node.parent_id
            if (parent_id != ""){
              ancestor = tree((parent_id, i))
              subVertices = ancestor.children_ids diff List(node.id)
              if (subVertices.length != 0){
                for (vertex <- subVertices){
                  neighbors = neighbors ++ find_all_children_ids(vertex, tree, i-1)
                }
              }
              if (neighbors.size != 0){
                for (neighbor <- neighbors){
                  if (sim.contains(neighbor)){
                    val value = sim(neighbor)+simNeighbor
                    sim += neighbor -> value
                  }else{
                    sim += neighbor -> simNeighbor
                  }
                }
              }
              node = ancestor
            }
          }
        }
      }
      if (topItems contains album){
        if (sim.keys.size !=0){
         var simTopItems = Map.empty[String, Double]  
          for (each <- topItems){
            if (sim.contains(each)){
              simTopItems += each -> sim(each)/numberOfTrees.toDouble
            }else{
              simTopItems += each -> 0.0
            }
          }
          val scores = simTopItems.toList.map(s => s._1+","+s._2.toString).mkString(",")
          (List(album) ++ sim.toList.sortWith((s,t) => s._2 > t._2).map(s => s._1).take(topK), album+","+scores)
        }else{
          (List(album), album+","+topItems.map(s => s+","+"0.0").mkString(","))
        }
      }else{
        if (sim.keys.size !=0){
          (List(album) ++ sim.toList.sortWith((s,t) => s._2 > t._2).map(s => s._1).take(topK),"not found")
        }else{
          (List(album),"not found")
        }
      }
    }
    
    val topSeachItem = als.map(s => query_items(s, trees, 50, topKitem))
    val topSearch = topSeachItem.map(s => s._1)
    val items = topSeachItem.map(s => s._2).filter(s => s != "not found")
    
    sc.parallelize(topSearch).saveAsTextFile("hdfs://master1.hh:8020/tmp/sim_rank/"+albumOrTag+"_"+treeDepth.toString+"_"+numberOfTrees.toString)
    sc.parallelize(items).saveAsTextFile("hdfs://master1.hh:8020/tmp/sim_rank/sim_"+albumOrTag+"_"+treeDepth.toString+"_"+numberOfTrees.toString)
   }    
   rw(10, 10, "tag")   
   rw(20, 20, "tag")  
 } 
}

