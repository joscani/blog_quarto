require(data.tree)

NodesInfo <-  function(arbol_ind){
  for (i in 1:length(arbol_ind)) {
    info <-
      sprintf(
        "Node ID %s has left child node with index %s and right child node with index %s The split feature is %s. The NA direction is %s",
        arbol_ind@node_ids[i],
        arbol_ind@left_children[i],
        arbol_ind@right_children[i],
        arbol_ind@features[i], 
        arbol_ind@nas[i]
      )
    print(info)
  }}
createDataTree <- function(h2oTree) {
  
  h2oTreeRoot = h2oTree@root_node
  
  dataTree = Node$new(h2oTreeRoot@split_feature)
  dataTree$type = 'split'
  
  addChildren(dataTree, h2oTreeRoot)
  
  return(dataTree)
}

addChildren <- function(dtree, node) {
  
  if(class(node)[1] != 'H2OSplitNode') return(TRUE)
  
  feature = node@split_feature
  id = node@id
  na_direction = node@na_direction
  
  if(is.na(node@threshold)) {
    leftEdgeLabel = printValues(node@left_levels, na_direction=='LEFT', 4)
    rightEdgeLabel = printValues(node@right_levels, na_direction=='RIGHT', 4)
  }else {
    leftEdgeLabel = paste("<", node@threshold, ifelse(na_direction=='LEFT',',NA',''))
    rightEdgeLabel = paste(">=", node@threshold, ifelse(na_direction=='RIGHT',',NA',''))
  }
  
  left_node = node@left_child
  right_node = node@right_child
  
  if(class(left_node)[[1]] == 'H2OLeafNode')
    leftLabel = paste("prediction:", left_node@prediction)
  else
    leftLabel = left_node@split_feature
  
  if(class(right_node)[[1]] == 'H2OLeafNode')
    rightLabel = paste("prediction:", right_node@prediction)
  else
    rightLabel = right_node@split_feature
  
  if(leftLabel == rightLabel) {
    leftLabel = paste(leftLabel, "(L)")
    rightLabel = paste(rightLabel, "(R)")
  }
  
  dtreeLeft = dtree$AddChild(leftLabel)
  dtreeLeft$edgeLabel = leftEdgeLabel
  dtreeLeft$type = ifelse(class(left_node)[1] == 'H2OSplitNode', 'split', 'leaf')
  
  dtreeRight = dtree$AddChild(rightLabel)
  dtreeRight$edgeLabel = rightEdgeLabel
  dtreeRight$type = ifelse(class(right_node)[1] == 'H2OSplitNode', 'split', 'leaf')
  
  addChildren(dtreeLeft, left_node)
  addChildren(dtreeRight, right_node)
  
  return(FALSE)
}

printValues <- function(values, is_na_direction, n=4) {
  l = length(values)
  
  if(l == 0)
    value_string = ifelse(is_na_direction, "NA", "")
  else
    value_string = paste0(paste0(values[1:min(n,l)], collapse = ', '),
                          ifelse(l > n, ",...", ""),
                          ifelse(is_na_direction, ", NA", ""))
  
  return(value_string)
}

plotDataTree <-  function(h2oDataTree, rankdir = "LR"){
  GetEdgeLabel <- function(node) {return (node$edgeLabel)}
  GetNodeShape <- function(node) {switch(node$type, split = "diamond", leaf = "oval")}
  GetFontName <- function(node) {switch(node$type, split = 'Palatino-bold', leaf = 'Palatino')}
  SetEdgeStyle(h2oDataTree, fontname = 'Palatino-italic', label = GetEdgeLabel, labelfloat = TRUE,
               fontsize = "26", fontcolor='royalblue4')
  SetNodeStyle(h2oDataTree, fontname = GetFontName, shape = GetNodeShape, 
               fontsize = "26", fontcolor='royalblue4',
               height="0.75", width="1")
  
  SetGraphStyle(h2oDataTree, rankdir = rankdir, dpi=70.)
  
  plot(h2oDataTree, output = "graph")
  
  
}
